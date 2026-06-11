# llm_lab

基于 Unsloth Docker 容器的 LLM 训练实验场，支持 SFT、GRPO 和多轮交互式 GRPO。

## 前置

- Docker Desktop（Windows）或 Docker + nvidia-container-toolkit（Linux）
- NVIDIA GPU（显存 >= 8GB 推荐）

## 快速开始

```bash
# 构建并启动
docker compose up -d

# 运行 SFT 训练
docker compose exec unsloth python scripts/train.py -c configs/example.yaml

# 运行多轮 GRPO 训练（基于 reasoning-gym）
docker compose exec -e HF_HUB_OFFLINE=1 unsloth python scripts/train.py -c configs/grpo_reasoning.yaml
```

## 项目结构

```
├── scripts/train.py          # 训练入口
├── configs/                  # YAML 训练配置
├── llm_lab/
│   ├── models/loader.py      # 模型加载（HF / ModelScope）
│   ├── data/
│   │   ├── dataset.py        # 数据集加载
│   │   └── reasoning.py      # reasoning-gym 适配
│   ├── environments/
│   │   ├── base.py           # 多轮环境抽象基类
│   │   └── reasoning_env.py  # reasoning-gym 多轮环境
│   └── trainers/
│       ├── sft.py            # SFT 训练
│       ├── grpo.py           # 标准 GRPO
│       ├── grpo_reasoning.py # 单轮 GRPO + reasoning
│       └── grpo_multi_turn.py# 多轮交互式 GRPO
├── data/                     # 训练数据（已忽略 git）
├── outputs/                  # 模型输出（已忽略 git）
└── models/                   # 本地模型缓存（已忽略 git）
```

## 支持的算法

| 算法 | 配置 |
|---|---|
| `sft` | 标准监督微调 |
| `grpo` | 标准 GRPO 强化学习 |
| `grpo_reasoning` | 单轮 GRPO + reasoning-gym |
| `grpo_multi_turn` | 多轮交互式 GRPO（模型与 environment 多步交互后获得 reward） |

## 多轮 GRPO

多轮 GRPO 允许模型在一个样本上与环境进行多步交互：

1. 模型生成推理 / 答案
2. 环境检查答案，给出反馈
3. 模型根据反馈继续推理
4. 重复直到给出正确答案或达到最大轮次

多轮训练的配置示例：

```yaml
training:
  algorithm: grpo_multi_turn
  num_generations: 2    # 每个 prompt 生成多少条轨迹
  beta: 0.04            # KL 惩罚系数
  reasoning:
    dataset_name: chain_sum
    max_turns: 3         # 最大交互轮次
    max_completion_length: 128
```

## 模型加载

支持 HuggingFace 和 ModelScope：

```yaml
model:
  source: modelscope          # huggingface | modelscope
  name_or_path: ./models/unsloth/Qwen3___5-0___8B
```

代码修改后即时生效，容器内 `/workspace` 映射项目根目录。