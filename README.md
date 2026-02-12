# MicroGPT - 最小化 GPT 实现 (Java)

一个使用**纯 Java** 从零实现的最小化 GPT 模型，无任何外部依赖。完整覆盖 **Pre-training → PPO Post-training** 全流程，适合学习和理解 GPT 与 RLHF 的工作原理。

## 核心特性

- **零依赖**：纯 Java 11 实现，没有 PyTorch / TensorFlow 等框架黑盒，每一步计算都透明可见
- **完整的自动微分引擎**：标量级别的 autograd，支持 `add / mul / pow / exp / log / relu / sigmoid / sub / div` 等运算及反向传播
- **标准 GPT 架构**：Token/Position Embedding → RMS Norm → Multi-Head Attention → MLP → 残差连接
- **KV Cache**：推理阶段使用 KV Cache 优化，与生产级实现一致
- **梯度累积**：支持 mini-batch 梯度累积，模拟更大的批次训练
- **Adam 优化器**：带余弦学习率衰减的 Adam 优化器
- **PPO 后训练**：完整的 PPO-Clip 强化学习训练器，包含 Reference Model、KL 散度惩罚、优势估计、Clipped Surrogate Objective
- **奖励函数**：基于规则的多维度奖励函数，评估生成质量（长度、元音比例、辅音-元音交替、新颖性等）
- **温度采样**：支持温度参数控制生成多样性

## 训练流程

```
┌──────────────────────────────────────────────────────────────────┐
│                     MicroGPT 完整训练流程                         │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │  Pre-train  │───▶│  PPO Post-   │───▶│  推理生成 (Inference)│  │
│  │  (预训练)    │    │  train (后训练)│    │                     │  │
│  └─────────────┘    └──────────────┘    └─────────────────────┘  │
│   · 交叉熵损失       · Clipped Surrogate  · 温度采样              │
│   · 梯度累积         · KL 散度惩罚         · KV Cache 加速        │
│   · 余弦学习率衰减    · Reference Model                          │
│                      · 优势标准化                                 │
└──────────────────────────────────────────────────────────────────┘
```

## 架构概览

```
输入 Token
    │
    ▼
┌─────────────────────────┐
│  Token Embedding (wte)  │
│  + Position Embedding   │
│       (wpe)             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      RMS Norm           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│     Transformer Block (× N_LAYER)  │
│  ┌───────────────────────────────┐  │
│  │  Multi-Head Self-Attention    │  │
│  │  (Q, K, V 投影 + KV Cache)   │  │
│  └──────────────┬────────────────┘  │
│                 │ + 残差连接         │
│  ┌──────────────▼────────────────┐  │
│  │  MLP (fc1 → ReLU² → fc2)     │  │
│  └──────────────┬────────────────┘  │
│                 │ + 残差连接         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────┐
│  Language Model Head    │
│     (lm_head)           │
└───────────┬─────────────┘
            │
            ▼
        输出 Logits
```

## PPO 后训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPO-Clip 算法流程                              │
│                                                                  │
│  1. Rollout 收集                                                 │
│     Policy Model ──生成序列──▶ [token₁, token₂, ..., tokenₙ]    │
│     Reference Model ──计算 log π_ref──▶ KL 约束基准              │
│                                                                  │
│  2. 奖励计算                                                     │
│     RewardFunction.score(生成文本) ──▶ reward                    │
│                                                                  │
│  3. 优势估计                                                     │
│     Aᵢ = rewardᵢ - baseline (batch 均值)                        │
│     标准化: Aᵢ = (Aᵢ - μ) / σ                                   │
│                                                                  │
│  4. 策略更新                                                     │
│     rₜ = π_new(aₜ|sₜ) / π_old(aₜ|sₜ)                          │
│     L = -min(rₜ·Aₜ, clip(rₜ, 1-ε, 1+ε)·Aₜ) + β·KL(π‖π_ref)  │
│                                                                  │
│  5. 定期同步 Reference Model                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
src/main/java/io/leavesfly/microgpt/
├── MicroGPT.java          # 主程序：串联 Pre-train → PPO Post-train → Inference 全流程
├── GPT.java               # GPT 模型：Transformer 架构，支持 cloneModel / syncParams
├── Value.java             # 自动微分引擎：标量计算图 + 反向传播
├── Tokenizer.java         # 字符级分词器
├── AdamOptimizer.java     # Adam 优化器（余弦学习率衰减）
├── PPOTrainer.java        # PPO-Clip 训练器：Rollout 收集、优势估计、策略更新
└── RewardFunction.java    # 奖励函数：多维度规则评估生成质量
```

### 各模块职责

| 模块 | 职责 | 核心概念 |
|------|------|----------|
| **Value** | 自动微分引擎，记录计算图并支持反向传播 | 计算图、链式法则、梯度累积、detach |
| **GPT** | Transformer 模型架构，支持模型克隆与参数同步 | Embedding、Multi-Head Attention、MLP、RMS Norm、残差连接、KV Cache |
| **Tokenizer** | 字符级分词，将文本转换为 token 序列 | 编码/解码、BOS 标记、词表构建 |
| **AdamOptimizer** | Adam 优化算法，自适应学习率 | 一阶/二阶矩估计、偏差修正、余弦学习率衰减 |
| **PPOTrainer** | PPO-Clip 强化学习训练器 | Rollout 收集、Clipped Surrogate Objective、KL 惩罚、优势标准化、Reference Model |
| **RewardFunction** | 基于规则的奖励函数 | 长度奖励、元音比例、辅音-元音交替、重复惩罚、新颖性奖励 |
| **MicroGPT** | 主程序，串联完整训练和推理流程 | Pre-train、PPO Post-train、温度采样 |

## 快速开始

### 环境要求

- Java 11+
- Maven 3.6+

### 编译运行

```bash
# 编译
mvn clean package

# 运行
java -jar target/microgpt-1.0.0.jar

# 或使用 Maven 直接运行
mvn exec:java
```

### 预期输出

```
========================================
    MicroGPT - 最小化 GPT 实现 (Java)
========================================

--- 加载数据集 ---
文档数量: 32033

--- 构建分词器 ---
词表大小: 27

--- 初始化模型 ---
模型参数数量: 5765

--- 模型配置 ---
嵌入维度: 16
注意力头数: 4
Transformer 层数: 1
最大序列长度: 8
参数数量: 5765

--- 训练前推理（未训练的随机输出）---
sample  1: xqkfwjm
sample  2: gtzblnr
...

--- 开始训练 ---
训练步数: 200, 批次大小: 4, 学习率: 0.0300

step    1 /  200 | loss 3.2958 | smooth_loss 3.2958
step   20 /  200 | loss 2.4521 | smooth_loss 2.5012
...
step  200 /  200 | loss 1.8234 | smooth_loss 1.8567

训练完成！(耗时: xxs)

--- 训练后推理 ---
温度参数: 0.60

sample  1: emma
sample  2: olivia
...

--- PPO 后训练 ---
PPO 步数: 200, Rollouts/步: 8, Clip ε: 0.20, KL β: 0.05, 学习率: 0.0010

ppo_step    1 /  200 | avg_reward 1.2345 | smooth_reward 1.2345 | ppo_loss 0.0123
ppo_step   10 /  200 | avg_reward 2.1234 | smooth_reward 1.8765 | ppo_loss 0.0089
...
ppo_step  200 /  200 | avg_reward 2.8901 | smooth_reward 2.7654 | ppo_loss 0.0034

PPO 后训练完成！(耗时: xxs)

--- PPO 后训练推理 ---
温度参数: 0.60

sample  1: elena        (reward: 3.20)
sample  2: maria        (reward: 2.90)
...
```

- **训练前**：模型输出随机字符组合
- **预训练后**：模型学会生成类似英文名字的字符序列
- **PPO 后训练后**：模型生成的名字质量进一步提升（更合理的长度、元音比例、辅音-元音交替模式，以及更高的新颖性）

## 模型配置

### 预训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_EMBD` | 16 | 嵌入维度 |
| `N_HEAD` | 4 | 注意力头数量 |
| `N_LAYER` | 1 | Transformer 层数 |
| `BLOCK_SIZE` | 8 | 最大序列长度 |
| `LEARNING_RATE` | 0.03 | 学习率 |
| `NUM_STEPS` | 200 | 训练步数 |
| `BATCH_SIZE` | 4 | 梯度累积批次大小 |
| `TEMPERATURE` | 0.6 | 推理温度（越低越确定，越高越多样） |

### PPO 后训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PPO_STEPS` | 200 | PPO 训练步数 |
| `PPO_ROLLOUTS_PER_STEP` | 8 | 每步生成的 rollout 数量 |
| `PPO_CLIP_EPSILON` | 0.2 | Clip 范围 ε，控制每步更新幅度 |
| `PPO_KL_COEFFICIENT` | 0.05 | KL 散度惩罚系数 β |
| `PPO_LEARNING_RATE` | 0.001 | PPO 学习率 |
| `PPO_TEMPERATURE` | 0.8 | PPO 推理温度 |
| `PPO_REF_SYNC_INTERVAL` | 25 | Reference Model 同步间隔 |

## 奖励函数设计

`RewardFunction` 从多个维度评估生成的英文名字质量：

| 维度 | 规则 | 分值范围 |
|------|------|----------|
| **长度** | 3-8 个字符得正分，过短或过长扣分 | -1.0 ~ 1.0 |
| **元音比例** | 元音占比 30%-60% 为最佳 | -0.5 ~ 0.8 |
| **首字母** | 常见英文名首字母加分 | 0.0 ~ 0.3 |
| **重复惩罚** | 连续 3+ 个相同字符扣分 | -1.0 ~ 0.3 |
| **新颖性** | 不在训练集中出现过的名字加分 | 0.0 ~ 0.8 |
| **辅音-元音交替** | 自然的辅音-元音交替模式加分 | 0.0 ~ 0.8 |

## 数据集

使用 [Andrej Karpathy 的 names.txt](https://github.com/karpathy/makemore)，包含约 32000 个英文名字。模型通过学习这些名字的字符模式，生成新的类似名字。PPO 阶段利用奖励函数进一步引导模型生成更高质量的名字。

## 致谢

本项目灵感来源于 [Andrej Karpathy](https://github.com/karpathy) 的 micrograd 和 makemore 系列教程，并在此基础上扩展了 PPO 强化学习后训练流程。
