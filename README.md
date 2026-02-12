# MicroGPT - 最小化 GPT 实现 (Java)

一个使用**纯 Java** 从零实现的最小化 GPT 模型，无任何外部依赖。完整覆盖 GPT 的核心技术栈，适合学习和理解 GPT 的工作原理。

## 核心特性

- **零依赖**：纯 Java 11 实现，没有 PyTorch / TensorFlow 等框架黑盒，每一步计算都透明可见
- **完整的自动微分引擎**：标量级别的 autograd，支持前向计算和反向传播
- **标准 GPT 架构**：Token/Position Embedding → Multi-Head Attention → MLP → 残差连接
- **KV Cache**：推理阶段使用 KV Cache 优化，与生产级实现一致
- **Adam 优化器**：带学习率线性衰减的 Adam 优化器
- **温度采样**：支持温度参数控制生成多样性

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

## 项目结构

```
src/main/java/io/leavesfly/microgpt/
├── MicroGPT.java        # 主程序：训练和推理流程
├── GPT.java              # GPT 模型：Transformer 架构实现
├── Value.java            # 自动微分引擎：标量计算图 + 反向传播
├── Tokenizer.java        # 字符级分词器
└── AdamOptimizer.java    # Adam 优化器
```

### 各模块职责

| 模块 | 职责 | 核心概念 |
|------|------|----------|
| **Value** | 自动微分引擎，记录计算图并支持反向传播 | 计算图、链式法则、梯度累积 |
| **GPT** | Transformer 模型架构 | Embedding、Multi-Head Attention、MLP、RMS Norm、残差连接 |
| **Tokenizer** | 字符级分词，将文本转换为 token 序列 | 编码/解码、BOS 标记、词表构建 |
| **AdamOptimizer** | Adam 优化算法，自适应学习率 | 一阶/二阶矩估计、偏差修正、学习率衰减 |
| **MicroGPT** | 主程序，串联训练和推理流程 | 训练循环、损失计算、温度采样 |

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
训练步数: 500, 学习率: 0.0100

step    1 / 500 | loss 3.2958
step   50 / 500 | loss 2.4521
step  100 / 500 | loss 2.1037
...
step  500 / 500 | loss 1.8234

训练完成！(耗时: 12.3s)

--- 训练后推理 ---
温度参数: 0.60

sample  1: emma
sample  2: olivia
sample  3: ava
...
```

训练前模型输出的是随机字符组合，训练后模型学会了生成类似英文名字的字符序列。

## 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_EMBD` | 16 | 嵌入维度 |
| `N_HEAD` | 4 | 注意力头数量 |
| `N_LAYER` | 1 | Transformer 层数 |
| `BLOCK_SIZE` | 8 | 最大序列长度 |
| `LEARNING_RATE` | 0.01 | 学习率 |
| `NUM_STEPS` | 500 | 训练步数 |
| `TEMPERATURE` | 0.6 | 推理温度（越低越确定，越高越多样） |

## 数据集

使用 [Andrej Karpathy 的 names.txt](https://github.com/karpathy/makemore)，包含约 32000 个英文名字。模型通过学习这些名字的字符模式，生成新的类似名字。

## 致谢

本项目灵感来源于 [Andrej Karpathy](https://github.com/karpathy) 的 micrograd 和 makemore 系列教程。
