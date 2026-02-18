## MicroGPT — 最小化 GPT 实现（纯 Java，零依赖）

从零实现一个完整的 GPT 语言模型，涵盖 **预训练 → REINFORCE 强化学习后训练 → 推理生成** 全流程。纯 Java 编写，无任何外部依赖，适合深入理解大语言模型的核心原理。

### 核心特性

- **自动微分引擎**：标量级别的计算图 + 反向传播，支持 `add`、`mul`、`exp`、`log`、`relu`、`pow` 等运算
- **GPT-2 风格架构**：Token/Position Embedding → Multi-Head Self-Attention → MLP → RMS Norm → 残差连接
- **KV Cache 推理优化**：逐 token 生成时复用历史 Key/Value，避免重复计算
- **Adam 优化器**：一阶/二阶矩估计 + 偏差修正 + 余弦学习率衰减
- **REINFORCE 后训练**：策略梯度 + 基线方差缩减 + KL 散度惩罚 + 梯度裁剪 + Advantage 裁剪
- **字符级分词器**：自动从训练数据构建词表

### 项目结构

```
src/main/java/io/leavesfly/microgpt/
├── Value.java          # 自动微分引擎（标量计算图 + 反向传播）
├── GPT.java            # GPT 模型（Transformer 架构 + 前向传播）
├── AdamOptimizer.java  # Adam 优化器（动量 + 自适应学习率）
├── Tokenizer.java      # 字符级分词器（编码/解码 + 数据加载）
└── MicroGPT.java       # 主程序（预训练 + RL 后训练 + 推理）

src/main/resources/
└── input.txt           # 训练数据集
```

### 训练流程

```
1. 加载数据集 & 构建分词器
2. 初始化 GPT 模型
3. 预训练（交叉熵损失 + 梯度累积 + Adam）
4. REINFORCE 后训练（策略梯度 + 奖励函数优化）
5. 推理生成（温度采样）
```

### 快速运行

```bash
# 编译
mvn compile

# 运行
mvn exec:java

# 或打包后运行
mvn package
java -jar target/microgpt-1.0.0.jar
```

### 模型配置

| 参数 | 值 | 说明 |
|---|---|---|
| 嵌入维度 | 16 | Token 的向量表示维度 |
| 注意力头数 | 4 | Multi-Head Attention 的头数 |
| Transformer 层数 | 1 | 堆叠的 Transformer Block 数量 |
| 最大序列长度 | 8 | 上下文窗口大小 |
| 预训练步数 | 100 | 交叉熵预训练迭代次数 |
| 预训练学习率 | 3e-2 | Adam 优化器学习率 |
| 批次大小 | 4 | 梯度累积的样本数 |

### REINFORCE 后训练

预训练完成后，使用 REINFORCE 策略梯度算法进一步优化模型生成质量：

**核心公式：** `loss = -(reward - baseline) × Σ log P(token_t) + λ × KL(π_current || π_ref)`

**奖励函数**（多维度综合评分）：
- **长度奖励**（30%）：鼓励生成接近最大长度的序列
- **多样性奖励**（30%）：唯一字符比例越高越好
- **合法性奖励**（20%）：字母和空格的占比
- **连续性奖励**（20%）：惩罚连续重复字符

**训练稳定性保障：**
- **基线方差缩减**：用批内奖励均值作为基线，降低梯度方差
- **KL 散度惩罚**（λ=0.3）：防止策略偏离预训练分布
- **L2 梯度裁剪**（max_norm=1.0）：防止梯度爆炸
- **Advantage 裁剪**（±0.5）：限制单条轨迹的影响力

| RL 参数 | 值 | 说明 |
|---|---|---|
| RL 步数 | 50 | 策略梯度更新次数 |
| 采样数 | 8 | 每步采样的轨迹数量 |
| RL 学习率 | 5e-4 | 比预训练小一个量级 |
| KL 惩罚系数 | 0.3 | 约束策略漂移 |
| 梯度裁剪 | 1.0 | L2 范数上限 |
| Advantage 裁剪 | ±0.5 | 限制异常轨迹影响 |

### 算法原理

#### 自动微分（Value.java）

每个 `Value` 节点记录数据值、梯度和反向传播函数，构成有向无环计算图。调用 `backward()` 时按拓扑逆序应用链式法则，自动计算所有参数的梯度。

#### Transformer 前向传播（GPT.java）

```
Input Token → Token Embedding + Position Embedding → RMS Norm
    → [Multi-Head Self-Attention + Residual] × N_LAYER
    → [MLP (FC → ReLU² → FC) + Residual] × N_LAYER
    → Linear → Logits
```

#### REINFORCE 策略梯度（MicroGPT.java）

```
for each RL step:
    1. 采样 8 条轨迹，记录 log P(token_t)
    2. 奖励函数打分，计算基线（均值）
    3. advantage = reward - baseline（裁剪到 ±0.5）
    4. loss = -advantage × Σ log P + KL_coeff × KL(π || π_ref)
    5. backward() → 梯度裁剪 → Adam 更新
```

### 环境要求

- **Java** 11+
- **Maven** 3.6+
- 无其他外部依赖
