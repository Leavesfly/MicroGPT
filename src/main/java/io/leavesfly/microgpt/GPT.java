package io.leavesfly.microgpt;

import java.util.*;

/**
 * GPT 类 - 生成式预训练 Transformer 模型
 * 
 * 实现 GPT-2 风格的架构，包含：
 * - Token Embedding 和 Position Embedding
 * - 多头自注意力机制 (Multi-Head Self-Attention)
 * - MLP 前馈网络
 * - RMS Norm 归一化
 * 
 */
public class GPT {
    // ============ 模型超参数 ============
    
    /** 嵌入维度 */
    private int nEmbd;
    
    /** 注意力头数量 */
    private int nHead;
    
    /** Transformer 层数 */
    private int nLayer;
    
    /** 最大序列长度 */
    private int blockSize;
    
    /** 每个注意力头的维度 */
    private int headDim;
    
    /** 词表大小 */
    private int vocabSize;
    
    // ============ 模型参数 ============
    
    /** 状态字典，存储所有模型参数（使用 LinkedHashMap 保证参数顺序确定性） */
    private Map<String, Value[][]> stateDict;
    
    /** 扁平化的参数列表，用于优化器更新 */
    private List<Value> params;
    
    // ============ 随机数生成器 ============
    private Random random;
    
    /**
     * 构造函数
     * @param vocabSize 词表大小
     * @param nEmbd 嵌入维度
     * @param nHead 注意力头数量
     * @param nLayer Transformer 层数
     * @param blockSize 最大序列长度
     */
    public GPT(int vocabSize, int nEmbd, int nHead, int nLayer, int blockSize) {
        this(vocabSize, nEmbd, nHead, nLayer, blockSize, true);
    }

    /**
     * 构造函数（支持静默模式）
     * @param vocabSize 词表大小
     * @param nEmbd 嵌入维度
     * @param nHead 注意力头数量
     * @param nLayer Transformer 层数
     * @param blockSize 最大序列长度
     * @param verbose 是否打印初始化信息
     */
    public GPT(int vocabSize, int nEmbd, int nHead, int nLayer, int blockSize, boolean verbose) {
        this.vocabSize = vocabSize;
        this.nEmbd = nEmbd;
        this.nHead = nHead;
        this.nLayer = nLayer;
        this.blockSize = blockSize;
        this.headDim = nEmbd / nHead;
        this.random = new Random(42);
        
        // 初始化模型参数
        initializeParameters();
        
        if (verbose) {
            System.out.println("模型参数数量: " + params.size());
        }
    }
    
    /**
     * 初始化模型参数
     */
    private void initializeParameters() {
        stateDict = new LinkedHashMap<>();
        
        // Token Embedding: wte[vocabSize][nEmbd]
        stateDict.put("wte", createMatrix(vocabSize, nEmbd, 0.02));
        
        // Position Embedding: wpe[blockSize][nEmbd]
        stateDict.put("wpe", createMatrix(blockSize, nEmbd, 0.02));
        
        // Language Model Head: lm_head[vocabSize][nEmbd]
        stateDict.put("lm_head", createMatrix(vocabSize, nEmbd, 0.02));
        
        // 为每一层创建参数
        for (int i = 0; i < nLayer; i++) {
            // Attention Q, K, V 权重
            stateDict.put("layer" + i + ".attn_wq", createMatrix(nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".attn_wk", createMatrix(nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".attn_wv", createMatrix(nEmbd, nEmbd, 0.02));
            
            // Attention Output 权重（初始化为0，类似于残差连接的初始化）
            stateDict.put("layer" + i + ".attn_wo", createMatrix(nEmbd, nEmbd, 0.0));
            
            // MLP 全连接层
            stateDict.put("layer" + i + ".mlp_fc1", createMatrix(4 * nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".mlp_fc2", createMatrix(nEmbd, 4 * nEmbd, 0.0));
        }
        
        // 将所有参数扁平化到单一列表
        params = new ArrayList<>();
        for (Value[][] matrix : stateDict.values()) {
            for (Value[] row : matrix) {
                Collections.addAll(params, row);
            }
        }
    }
    
    /**
     * 创建参数矩阵
     * @param nOut 输出维度
     * @param nIn 输入维度
     * @param std 初始化标准差
     * @return 参数矩阵
     */
    private Value[][] createMatrix(int nOut, int nIn, double std) {
        Value[][] matrix = new Value[nOut][nIn];
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                // 使用高斯分布初始化
                double val = random.nextGaussian() * std;
                matrix[i][j] = new Value(val);
            }
        }
        return matrix;
    }
    
    // ============ 核心运算函数 ============
    
    /**
     * 线性变换（矩阵乘法）
     * y = x @ W^T，其中 W 是权重矩阵
     * @param x 输入向量
     * @param w 权重矩阵
     * @return 输出向量
     */
    public Value[] linear(Value[] x, Value[][] w) {
        int nOut = w.length;
        int nIn = w[0].length;
        
        Value[] out = new Value[nOut];
        for (int i = 0; i < nOut; i++) {
            Value sum = new Value(0);
            for (int j = 0; j < nIn; j++) {
                sum = sum.add(w[i][j].mul(x[j]));
            }
            out[i] = sum;
        }
        return out;
    }
    
    /**
     * Softmax 函数
     * 将向量转换为概率分布
     * @param logits 输入向量
     * @return 概率分布
     */
    public Value[] softmax(Value[] logits) {
        // 找最大值用于数值稳定性
        double maxVal = Double.NEGATIVE_INFINITY;
        for (Value v : logits) {
            if (v.data > maxVal) {
                maxVal = v.data;
            }
        }
        
        // 计算 exp(logits - max)
        Value[] exps = new Value[logits.length];
        Value sum = new Value(0);
        for (int i = 0; i < logits.length; i++) {
            exps[i] = logits[i].add(-maxVal).exp();
            sum = sum.add(exps[i]);
        }
        
        // 归一化
        Value[] probs = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = exps[i].div(sum);
        }
        return probs;
    }
    
    /**
     * RMS Norm 归一化
     * Root Mean Square Layer Normalization
     * @param x 输入向量
     * @return 归一化后的向量
     */
    public Value[] rmsnorm(Value[] x) {
        int n = x.length;
        
        // 计算均方值
        Value ms = new Value(0);
        for (Value xi : x) {
            ms = ms.add(xi.mul(xi));
        }
        ms = ms.div(n);
        
        // 计算缩放因子
        Value scale = ms.add(1e-5).pow(-0.5);
        
        // 应用缩放
        Value[] out = new Value[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[i].mul(scale);
        }
        return out;
    }
    
    // ============ GPT 前向传播 ============
    
    /**
     * GPT 前向传播
     * @param tokenId 当前 token 的索引
     * @param posId 当前位置的索引
     * @param keys KV Cache 中的 keys
     * @param values KV Cache 中的 values
     * @return logits 向量
     */
    public Value[] forward(int tokenId, int posId, List<List<Value[]>> keys, List<List<Value[]>> values) {
        // 1. Token Embedding + Position Embedding
        Value[] tokEmb = getEmbedding(stateDict.get("wte"), tokenId);
        Value[] posEmb = getEmbedding(stateDict.get("wpe"), posId);
        
        // 合并 token 和 position embedding
        Value[] x = new Value[nEmbd];
        for (int i = 0; i < nEmbd; i++) {
            x[i] = tokEmb[i].add(posEmb[i]);
        }
        
        // RMS Norm
        x = rmsnorm(x);
        
        // 2. Transformer 层
        for (int li = 0; li < nLayer; li++) {
            // ============ Multi-Head Attention Block ============
            Value[] xResidual = x;
            x = rmsnorm(x);
            
            // 计算 Q, K, V
            Value[] q = linear(x, stateDict.get("layer" + li + ".attn_wq"));
            Value[] k = linear(x, stateDict.get("layer" + li + ".attn_wk"));
            Value[] v = linear(x, stateDict.get("layer" + li + ".attn_wv"));
            
            // 将 K, V 添加到缓存
            keys.get(li).add(k);
            values.get(li).add(v);
            
            // 多头注意力计算（使用偏移量索引，避免数组拷贝）
            Value[] xAttn = new Value[nEmbd];
            List<Value[]> layerKeys = keys.get(li);
            List<Value[]> layerValues = values.get(li);
            int seqLen = layerKeys.size();
            double scaleFactor = Math.sqrt(headDim);

            for (int h = 0; h < nHead; h++) {
                int hs = h * headDim;

                // 计算注意力分数: Q @ K^T / sqrt(headDim)，直接用偏移量访问
                Value[] attnLogits = new Value[seqLen];
                for (int t = 0; t < seqLen; t++) {
                    Value[] kt = layerKeys.get(t);
                    Value score = new Value(0);
                    for (int j = 0; j < headDim; j++) {
                        score = score.add(q[hs + j].mul(kt[hs + j]));
                    }
                    attnLogits[t] = score.div(scaleFactor);
                }

                // Softmax 得到注意力权重
                Value[] attnWeights = softmax(attnLogits);

                // 加权求和得到输出，直接用偏移量访问
                for (int j = 0; j < headDim; j++) {
                    Value sum = new Value(0);
                    for (int t = 0; t < seqLen; t++) {
                        sum = sum.add(attnWeights[t].mul(layerValues.get(t)[hs + j]));
                    }
                    xAttn[hs + j] = sum;
                }
            }
            
            // Attention 输出投影
            x = linear(xAttn, stateDict.get("layer" + li + ".attn_wo"));
            
            // 残差连接
            for (int i = 0; i < nEmbd; i++) {
                x[i] = x[i].add(xResidual[i]);
            }
            
            // ============ MLP Block ============
            xResidual = x;
            x = rmsnorm(x);
            
            // MLP 第一层（扩展维度）
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc1"));
            
            // 激活函数: ReLU^2 (Square ReLU)
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].relu().pow(2);
            }
            
            // MLP 第二层（恢复维度）
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc2"));
            
            // 残差连接
            for (int i = 0; i < nEmbd; i++) {
                x[i] = x[i].add(xResidual[i]);
            }
        }
        
        // 3. 输出层
        Value[] logits = linear(x, stateDict.get("lm_head"));
        return logits;
    }
    
    /**
     * 从嵌入矩阵中获取指定索引的嵌入向量（直接返回引用，避免不必要的拷贝）
     */
    private Value[] getEmbedding(Value[][] matrix, int idx) {
        return matrix[idx];
    }
    
    // ============ Getter 方法 ============
    
    public List<Value> getParams() {
        return params;
    }
    
    public int getBlockSize() {
        return blockSize;
    }
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public int getNEmbd() {
        return nEmbd;
    }
    
    public int getNHead() {
        return nHead;
    }
    
    public int getNLayer() {
        return nLayer;
    }
    
    public Map<String, Value[][]> getStateDict() {
        return stateDict;
    }
    
    /**
     * 初始化 KV Cache
     * 用于存储注意力计算中的历史 K 和 V
     */
    public List<List<Value[]>> initKVCache() {
        List<List<Value[]>> cache = new ArrayList<>();
        for (int i = 0; i < nLayer; i++) {
            cache.add(new ArrayList<>());
        }
        return cache;
    }

    /**
     * 深拷贝模型（用于 PPO 中的 Reference Model）
     * 拷贝后的模型参数值相同，但拥有独立的 Value 节点，不共享计算图。
     *
     * @return 深拷贝后的 GPT 模型
     */
    public GPT cloneModel() {
        GPT cloned = new GPT(vocabSize, nEmbd, nHead, nLayer, blockSize, false);

        for (Map.Entry<String, Value[][]> entry : this.stateDict.entrySet()) {
            String key = entry.getKey();
            Value[][] srcMatrix = entry.getValue();
            Value[][] dstMatrix = cloned.stateDict.get(key);

            for (int i = 0; i < srcMatrix.length; i++) {
                for (int j = 0; j < srcMatrix[i].length; j++) {
                    dstMatrix[i][j] = new Value(srcMatrix[i][j].data);
                }
            }
        }

        // 重新构建扁平化参数列表
        cloned.params = new ArrayList<>();
        for (Value[][] matrix : cloned.stateDict.values()) {
            for (Value[] row : matrix) {
                Collections.addAll(cloned.params, row);
            }
        }

        return cloned;
    }

    /**
     * 从另一个模型同步参数值（用于 PPO 中定期更新 Reference Model）
     *
     * @param source 源模型
     */
    public void syncParamsFrom(GPT source) {
        List<Value> srcParams = source.getParams();
        for (int i = 0; i < this.params.size(); i++) {
            this.params.get(i).data = srcParams.get(i).data;
        }
    }
}
