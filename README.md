# llm_lab

基于 Unsloth Docker 容量的 LLM 训练实验场。

## 前置

- Docker Desktop（Windows）或 Docker + nvidia-container-toolkit（Linux）
- NVIDIA GPU（显存 >= 8GB 推荐）

## 快速开始

```bash
# 启动容器（后台）
docker compose up -d

# 验证环境
docker compose exec unsloth python -c "import unsloth; print(unsloth.__version__)"

# 运行训练
docker compose exec unsloth python scripts/train.py -c configs/example.yaml

# 停止
docker compose down
```

## 项目结构

```
├── scripts/          # 训练入口
├── configs/          # 训练配置
├── data/             # 训练数据（已忽略 git）
├── outputs/          # 模型输出（已忽略 git）
└── docker-compose.yml
```

代码修改后即时生效，容器内 `/workspace` 映射项目根目录。