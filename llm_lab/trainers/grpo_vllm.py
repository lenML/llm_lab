"""Multi-turn REINFORCE trainer with vLLM-accelerated generation.

Same REINFORCE + entropy algorithm as ``grpo_multi_turn`` but uses vLLM
for the generation (rollout) phase (100-300x faster for small models).

Usage::

    from vllm import LLM, SamplingParams

    vllm = LLM(model="...", ...)
    sp = SamplingParams(temperature=0.6, max_tokens=128)

    train_grpo_vllm(
        model=unsloth_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        vllm_model=vllm,
        vllm_sampling_params=sp,
        env_factory=...,
    )
"""

import os

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from llm_lab.environments import BaseEnvironment


def _tokenize(tokenizer, text, device, max_length=None):
    return tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_length or 4096,
    )["input_ids"].to(device)


def _compute_trajectory_log_probs(model, turns, device):
    total = torch.tensor(0.0, device=device)
    for prompt_ids, gen_ids in turns:
        gen_len = len(gen_ids)
        if gen_len == 0 or len(prompt_ids) == 0:
            continue
        prompt_len = len(prompt_ids)
        full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(device)
        outputs = model(full, attention_mask=torch.ones_like(full))
        logits = outputs.logits[0]
        gen_logits = logits[prompt_len - 1: prompt_len + gen_len - 1]
        gen_lp = torch.log_softmax(gen_logits, dim=-1)
        token_lp = gen_lp[torch.arange(gen_len, device=device), gen_ids]
        total = total + token_lp.sum()
    return total


def _compute_reinforce_loss(model, trajectories, entropy_coef, device):
    curr_log_probs = []
    for turns, _ in trajectories:
        lp = _compute_trajectory_log_probs(model, turns, device)
        curr_log_probs.append(lp)
    curr_lp = torch.stack(curr_log_probs)
    rewards = torch.tensor([r for _, r in trajectories], device=device, dtype=torch.float)
    if rewards.numel() > 1:
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        adv = torch.zeros_like(rewards)
    entropy = -curr_lp.mean()
    loss = -(adv.detach() * curr_lp).mean() - entropy_coef * entropy
    return loss, curr_lp.detach(), rewards.detach()


def train_grpo_vllm(
    model, tokenizer, train_dataset, *,
    vllm_model, vllm_sampling_params,
    env_factory,
    num_generations=4, max_turns=3,
    max_prompt_length=512, max_completion_length=128,
    entropy_coef=0.01, learning_rate=1e-6,
    batch_size=1, num_epochs=1,
    output_dir="./outputs/grpo_vllm",
    logging_steps=1, save_steps=100,
    gradient_accumulation_steps=1,
):
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)

    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(num_epochs):
        for batch_items in dataloader:
            # create envs for all items x generations
            envs = []
            for item in batch_items:
                for _ in range(num_generations):
                    env = env_factory()
                    env.reset(item)
                    envs.append(env)
            if not envs:
                continue

            trajectories = [[] for _ in envs]

            # multi-turn rollout with vLLM batch gen per turn
            for _turn in range(max_turns):
                active = [(i, env) for i, env in enumerate(envs) if not env.done]
                if not active:
                    break

                prompts = []
                for _, env in active:
                    raw = env.get_prompt()
                    truncated = tokenizer.decode(
                        tokenizer.encode(raw, truncation=True, max_length=max_prompt_length),
                        skip_special_tokens=True,
                    )
                    prompts.append(truncated)

                outputs = vllm_model.generate(prompts, vllm_sampling_params)

                for (idx, env), out in zip(active, outputs):
                    response = out.outputs[0].text
                    if not response.strip():
                        env.step("")
                        continue
                    prompt_text = env.get_prompt()
                    prompt_ids = _tokenize(tokenizer, prompt_text, device, max_prompt_length)[0]
                    full_text = prompt_text + response
                    full_ids = _tokenize(
                        tokenizer, full_text, device,
                        max_prompt_length + max_completion_length * 2,
                    )[0]
                    gen_ids = full_ids[prompt_ids.shape[0]:]
                    if len(gen_ids) == 0:
                        env.step("")
                        continue
                    env.step(response)
                    trajectories[idx].append((prompt_ids.detach().cpu(), gen_ids.detach().cpu()))

            rewards_list = [float(env.final_reward) for env in envs]
            traj_with_reward = [(trajectories[i], rewards_list[i]) for i in range(len(envs))]

            if not traj_with_reward:
                global_step += 1
                continue

            loss, curr_lp_log, rewards_log = _compute_reinforce_loss(
                model, traj_with_reward, entropy_coef, device
            )
            loss_scaled = loss / gradient_accumulation_steps
            loss_scaled.backward()
            accumulated_loss += loss_scaled.item()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % logging_steps == 0:
                mean_r = sum(rewards_list) / len(rewards_list)
                max_r = max(rewards_list)
                print(
                    f"Step {global_step:>5d} | loss {accumulated_loss:.4f} | "
                    f"reward mean {mean_r:.3f} max {max_r:.3f} | "
                    f"mean_lp {curr_lp_log.mean().item():.3f}"
                )
                accumulated_loss = 0.0

            if global_step % save_steps == 0:
                ckpt = os.path.join(output_dir, f"step_{global_step}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  -> saved checkpoint to {ckpt}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  v model saved to {output_dir}")
