package io.leavesfly.microgpt;

import java.util.*;

/**
 * MicroGPT 主程序 - 训练和推理 GPT 模型
 * <p>
 * 这是最小化的 GPT 训练和推理实现。
 * 使用纯 Java，无外部依赖，完整实现 GPT 算法。
 */
public class MicroGPT {

    // ============ 训练超参数 ============

    /**
     * 嵌入维度
     */
    private static final int N_EMBD = 16;

    /**
     * 注意力头数量
     */
    private static final int N_HEAD = 4;

    /**
     * Transformer 层数
     */
    private static final int N_LAYER = 1;

    /**
     * 最大序列长度
     */
    private static final int BLOCK_SIZE = 8;

    /**
     * 学习率
     */
    private static final double LEARNING_RATE = 3e-2;

    /**
     * 训练步数（每步包含 BATCH_SIZE 个样本的梯度累积）
     */
    private static final int NUM_STEPS = 100;

    /**
     * 梯度累积批次大小
     */
    private static final int BATCH_SIZE = 4;

    /**
     * 推理温度参数（控制生成多样性）
     */
    private static final double TEMPERATURE = 0.6;

    /**
     * 生成样本数量
     */
    private static final int NUM_SAMPLES = 20;

    // ============ REINFORCE 后训练超参数 ============

    /**
     * RL 训练步数
     */
    private static final int RL_STEPS = 50;

    /**
     * RL 每步采样的序列数量（用于估计基线和降低方差）
     */
    private static final int RL_SAMPLE_SIZE = 8;

    /**
     * RL 学习率（通常比预训练小）
     */
    private static final double RL_LEARNING_RATE = 5e-4;

    /**
     * KL 散度惩罚系数（防止策略偏离预训练分布太远）
     */
    private static final double KL_PENALTY_COEFF = 0.3;

    /**
     * 梯度裁剪阈值（防止异常轨迹导致的梯度爆炸）
     */
    private static final double GRAD_CLIP_NORM = 1.0;

    /**
     * Advantage 裁剪范围（限制单条轨迹的影响力）
     */
    private static final double ADVANTAGE_CLIP = 0.5;

    // ============ 核心组件 ============

    private Tokenizer tokenizer;
    private GPT model;
    private AdamOptimizer optimizer;
    private AdamOptimizer rlOptimizer;
    private List<String> docs;
    private Random random;

    /**
     * 主函数入口
     */
    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("    MicroGPT - 最小化 GPT 实现 (Java)");
        System.out.println("========================================\n");

        MicroGPT microGPT = new MicroGPT();
        microGPT.run();
    }

    /**
     * 运行完整的训练和推理流程
     */
    public void run() {
        // 设置随机种子，确保可复现性
        random = new Random(42);

        // 1. 加载数据集
        loadDataset();

        // 2. 构建分词器
        buildTokenizer();

        // 3. 初始化模型
        initializeModel();

        // 4. 打印模型配置
        printConfig();

        // 5. 训练前推理（展示未训练的随机输出）
        System.out.println("\n--- 训练前推理（未训练的随机输出）---");
        generateSamples(5);

        // 6. 训练模型
        train();

        // 7. REINFORCE 后训练（强化学习微调）
        reinforceFinetune();

        // 8. 训练后推理生成
        inference();
    }

    /**
     * 从 classpath resources 加载数据集
     */
    private void loadDataset() {
        System.out.println("--- 加载数据集 ---");

        docs = Tokenizer.loadDataset("input.txt");

        // 打乱数据顺序
        Collections.shuffle(docs, random);

        System.out.println("文档数量: " + docs.size());
    }

    /**
     * 构建分词器
     */
    private void buildTokenizer() {
        System.out.println("\n--- 构建分词器 ---");
        tokenizer = new Tokenizer(docs);
    }

    /**
     * 初始化模型和优化器
     */
    private void initializeModel() {
        System.out.println("\n--- 初始化模型 ---");

        model = new GPT(
                tokenizer.getVocabSize(),
                N_EMBD,
                N_HEAD,
                N_LAYER,
                BLOCK_SIZE
        );

        optimizer = new AdamOptimizer(
                LEARNING_RATE,
                0.9,    // beta1
                0.95,   // beta2
                1e-8,   // eps
                model.getParams().size()
        );
    }

    /**
     * 训练循环（使用梯度累积模拟 mini-batch）
     */
    private void train() {
        System.out.println("\n--- 开始训练 ---");
        System.out.printf("训练步数: %d, 批次大小: %d, 学习率: %.4f%n%n", NUM_STEPS, BATCH_SIZE, LEARNING_RATE);

        long trainStartTime = System.currentTimeMillis();
        int sampleIdx = 0;
        double smoothLoss = -1;

        for (int step = 0; step < NUM_STEPS; step++) {
            // 清零梯度
            AdamOptimizer.zeroGrad(model.getParams());

            double batchLoss = 0;

            // 梯度累积：对 BATCH_SIZE 个样本累积梯度
            for (int batchIdx = 0; batchIdx < BATCH_SIZE; batchIdx++) {
                String doc = docs.get(sampleIdx % docs.size());
                sampleIdx++;

                // Tokenize：添加 BOS 标记
                int[] tokens = new int[doc.length() + 2];
                tokens[0] = tokenizer.getBOS();
                for (int i = 0; i < doc.length(); i++) {
                    tokens[i + 1] = tokenizer.encode(doc.charAt(i));
                }
                tokens[doc.length() + 1] = tokenizer.getBOS();

                int n = Math.min(BLOCK_SIZE, tokens.length - 1);

                // 前向传播
                List<List<Value[]>> keys = model.initKVCache();
                List<List<Value[]>> values = model.initKVCache();
                List<Value> losses = new ArrayList<>();

                for (int posId = 0; posId < n; posId++) {
                    int tokenId = tokens[posId];
                    int targetId = tokens[posId + 1];

                    Value[] logits = model.forward(tokenId, posId, keys, values);
                    Value[] probs = model.softmax(logits);
                    Value lossT = probs[targetId].log().mul(-1);
                    losses.add(lossT);
                }

                // 计算该样本的平均损失，除以 BATCH_SIZE 实现梯度平均
                Value loss = new Value(0);
                for (Value l : losses) {
                    loss = loss.add(l);
                }
                loss = loss.div(n * BATCH_SIZE);

                // 反向传播（梯度会累积到参数上）
                loss.backward();

                batchLoss += loss.data * BATCH_SIZE;
            }

            // 使用 Adam 优化器更新参数
            optimizer.step(model.getParams(), step, NUM_STEPS);

            // 计算滑动平均 loss
            double avgLoss = batchLoss / BATCH_SIZE;
            smoothLoss = (smoothLoss < 0) ? avgLoss : 0.9 * smoothLoss + 0.1 * avgLoss;

            // 打印进度
            if ((step + 1) % 20 == 0 || step == 0) {
                System.out.printf("step %4d / %4d | loss %.4f | smooth_loss %.4f%n",
                        step + 1, NUM_STEPS, avgLoss, smoothLoss);
            }
        }

        long trainEndTime = System.currentTimeMillis();
        double trainSeconds = (trainEndTime - trainStartTime) / 1000.0;
        System.out.printf("%n训练完成！(耗时: %.1fs)%n", trainSeconds);
    }

    // ============ REINFORCE 后训练 ============

    /**
     * REINFORCE 后训练 - 使用策略梯度优化模型
     *
     * 核心思想：生成序列 → 奖励函数打分 → 策略梯度更新
     * loss = -(reward - baseline) × Σ log P(token_t)
     */
    private void reinforceFinetune() {
        System.out.println("\n--- REINFORCE 后训练 ---");
        System.out.printf("RL 步数: %d, 采样数: %d, 学习率: %.4f, KL 惩罚: %.2f, 梯度裁剪: %.1f%n%n",
                RL_STEPS, RL_SAMPLE_SIZE, RL_LEARNING_RATE, KL_PENALTY_COEFF, GRAD_CLIP_NORM);

        // 1. 保存预训练时的参考 logits（用于 KL 散度惩罚）
        // 用一组固定的 token 序列来计算参考分布
        double[][] referenceLogProbs = captureReferenceDistribution();

        // 2. 初始化 RL 专用优化器（较小学习率，重置动量状态）
        rlOptimizer = new AdamOptimizer(RL_LEARNING_RATE, 0.9, 0.95, 1e-8, model.getParams().size());

        long rlStartTime = System.currentTimeMillis();
        double smoothReward = -1;

        for (int step = 0; step < RL_STEPS; step++) {
            AdamOptimizer.zeroGrad(model.getParams());

            double totalReward = 0;
            double totalLossValue = 0;

            // 收集多个轨迹的奖励，用于计算基线
            double[] rewards = new double[RL_SAMPLE_SIZE];
            List<List<Value>> allLogProbs = new ArrayList<>();
            List<List<Integer>> allGeneratedTokens = new ArrayList<>();

            // 第一遍：采样生成序列并计算奖励
            for (int sampleIdx = 0; sampleIdx < RL_SAMPLE_SIZE; sampleIdx++) {
                List<List<Value[]>> keys = model.initKVCache();
                List<List<Value[]>> values = model.initKVCache();

                int tokenId = tokenizer.getBOS();
                List<Value> logProbs = new ArrayList<>();
                List<Integer> generatedTokens = new ArrayList<>();

                for (int posId = 0; posId < BLOCK_SIZE; posId++) {
                    Value[] logits = model.forward(tokenId, posId, keys, values);
                    Value[] probs = model.softmax(logits);

                    // 采样下一个 token
                    tokenId = sampleFromProbs(probs);

                    if (tokenId == tokenizer.getBOS()) {
                        break;
                    }

                    // 记录 log P(sampled_token) —— 保留在计算图中
                    logProbs.add(probs[tokenId].log());
                    generatedTokens.add(tokenId);
                }

                rewards[sampleIdx] = computeReward(generatedTokens);
                allLogProbs.add(logProbs);
                allGeneratedTokens.add(generatedTokens);
                totalReward += rewards[sampleIdx];
            }

            // 计算基线（奖励均值），用于降低方差
            double baseline = totalReward / RL_SAMPLE_SIZE;

            // 第二遍：计算策略梯度 loss 并反向传播
            for (int sampleIdx = 0; sampleIdx < RL_SAMPLE_SIZE; sampleIdx++) {
                List<Value> logProbs = allLogProbs.get(sampleIdx);
                if (logProbs.isEmpty()) {
                    continue;
                }

                double advantage = rewards[sampleIdx] - baseline;

                // 裁剪 advantage，防止异常轨迹主导梯度
                advantage = Math.max(-ADVANTAGE_CLIP, Math.min(ADVANTAGE_CLIP, advantage));

                // 策略梯度 loss = -(advantage) × Σ log P(token_t) / (序列长度 × 采样数)
                Value loss = new Value(0);
                for (Value logP : logProbs) {
                    loss = loss.add(logP.mul(-advantage));
                }

                // KL 散度惩罚：鼓励模型不要偏离预训练分布太远
                Value klPenalty = computeKLPenalty(allGeneratedTokens.get(sampleIdx),
                        allLogProbs.get(sampleIdx), referenceLogProbs);
                loss = loss.add(klPenalty.mul(KL_PENALTY_COEFF));

                loss = loss.div(logProbs.size() * RL_SAMPLE_SIZE);
                loss.backward();

                totalLossValue += loss.data * RL_SAMPLE_SIZE;
            }

            // 梯度裁剪：防止梯度爆炸
            clipGradNorm(model.getParams(), GRAD_CLIP_NORM);

            // 更新参数
            rlOptimizer.step(model.getParams(), step, RL_STEPS);

            // 统计
            double avgReward = totalReward / RL_SAMPLE_SIZE;
            smoothReward = (smoothReward < 0) ? avgReward : 0.9 * smoothReward + 0.1 * avgReward;

            if ((step + 1) % 10 == 0 || step == 0) {
                System.out.printf("rl_step %3d / %3d | reward %.4f | smooth_reward %.4f | loss %.4f%n",
                        step + 1, RL_STEPS, avgReward, smoothReward, totalLossValue / RL_SAMPLE_SIZE);
            }
        }

        long rlEndTime = System.currentTimeMillis();
        double rlSeconds = (rlEndTime - rlStartTime) / 1000.0;
        System.out.printf("%nREINFORCE 后训练完成！(耗时: %.1fs)%n", rlSeconds);

        // 展示 RL 后训练效果
        System.out.println("\n--- RL 后训练推理 ---");
        generateSamples(5);
    }

    /**
     * 计算奖励函数
     *
     * 综合多个维度对生成序列打分：
     * 1. 长度奖励：鼓励生成更长的有意义序列
     * 2. 多样性奖励：惩罚字符重复
     * 3. 合法性奖励：奖励生成合法的字母和空格组合
     *
     * @param generatedTokens 生成的 token 序列
     * @return 奖励值
     */
    private double computeReward(List<Integer> generatedTokens) {
        if (generatedTokens.isEmpty()) {
            return -1.0;
        }

        double reward = 0.0;

        // 解码为字符串
        StringBuilder sb = new StringBuilder();
        for (int tokenId : generatedTokens) {
            sb.append(tokenizer.decode(tokenId));
        }
        String generated = sb.toString();

        // 1. 长度奖励：鼓励生成接近 BLOCK_SIZE 的序列（归一化到 0~1）
        double lengthReward = Math.min(1.0, (double) generated.length() / BLOCK_SIZE);
        reward += lengthReward * 0.3;

        // 2. 多样性奖励：唯一字符比例（归一化到 0~1）
        Set<Character> uniqueChars = new HashSet<>();
        for (char c : generated.toCharArray()) {
            uniqueChars.add(c);
        }
        double diversityReward = (double) uniqueChars.size() / Math.max(1, generated.length());
        reward += diversityReward * 0.3;

        // 3. 合法性奖励：字母和空格的比例
        int validCharCount = 0;
        for (char c : generated.toCharArray()) {
            if (Character.isLetter(c) || c == ' ' || c == '\'' || c == '-') {
                validCharCount++;
            }
        }
        double validityReward = (double) validCharCount / Math.max(1, generated.length());
        reward += validityReward * 0.2;

        // 4. 连续性奖励：惩罚连续重复字符（如 "aaaa"）
        int repeatPenaltyCount = 0;
        for (int i = 1; i < generated.length(); i++) {
            if (generated.charAt(i) == generated.charAt(i - 1)) {
                repeatPenaltyCount++;
            }
        }
        double repeatPenalty = 1.0 - (double) repeatPenaltyCount / Math.max(1, generated.length() - 1);
        reward += repeatPenalty * 0.2;

        return reward;
    }

    /**
     * 捕获预训练模型的参考分布
     * 用固定的 BOS 起始，记录每个位置的 log 概率分布
     *
     * @return 参考 log 概率矩阵 [position][vocabSize]
     */
    private double[][] captureReferenceDistribution() {
        double[][] refLogProbs = new double[BLOCK_SIZE][model.getVocabSize()];

        List<List<Value[]>> keys = model.initKVCache();
        List<List<Value[]>> values = model.initKVCache();

        int tokenId = tokenizer.getBOS();
        for (int posId = 0; posId < BLOCK_SIZE; posId++) {
            Value[] logits = model.forward(tokenId, posId, keys, values);
            Value[] probs = model.softmax(logits);

            for (int v = 0; v < model.getVocabSize(); v++) {
                refLogProbs[posId][v] = Math.log(Math.max(probs[v].data, 1e-10));
            }

            // 用 argmax 选择下一个 token（确定性参考路径）
            tokenId = 0;
            double maxProb = -1;
            for (int v = 0; v < probs.length; v++) {
                if (probs[v].data > maxProb) {
                    maxProb = probs[v].data;
                    tokenId = v;
                }
            }
        }

        return refLogProbs;
    }

    /**
     * 计算 KL 散度惩罚
     * KL(π_current || π_ref) ≈ Σ [log π_current(a_t) - log π_ref(a_t)]
     *
     * @param generatedTokens 生成的 token 序列
     * @param currentLogProbs 当前策略的 log 概率（Value 类型，在计算图中）
     * @param referenceLogProbs 参考分布的 log 概率
     * @return KL 惩罚值（Value 类型，支持反向传播）
     */
    private Value computeKLPenalty(List<Integer> generatedTokens,
                                   List<Value> currentLogProbs,
                                   double[][] referenceLogProbs) {
        Value klDivergence = new Value(0);

        int length = Math.min(generatedTokens.size(), currentLogProbs.size());
        for (int t = 0; t < length && t < BLOCK_SIZE; t++) {
            int tokenId = generatedTokens.get(t);
            Value currentLogP = currentLogProbs.get(t);
            double refLogP = referenceLogProbs[t][tokenId];

            // KL ≈ log π_current - log π_ref
            klDivergence = klDivergence.add(currentLogP.sub(refLogP));
        }

        return klDivergence;
    }

    /**
     * 梯度裁剪（L2 范数裁剪）
     * 当梯度的 L2 范数超过阈值时，按比例缩小所有梯度
     *
     * @param params 模型参数列表
     * @param maxNorm 最大梯度范数
     */
    private void clipGradNorm(List<Value> params, double maxNorm) {
        double totalNormSquared = 0;
        for (Value p : params) {
            totalNormSquared += p.grad * p.grad;
        }
        double totalNorm = Math.sqrt(totalNormSquared);

        if (totalNorm > maxNorm) {
            double scale = maxNorm / totalNorm;
            for (Value p : params) {
                p.grad *= scale;
            }
        }
    }

    /**
     * 推理生成
     */
    private void inference() {
        System.out.println("\n--- 训练后推理 ---");
        System.out.printf("温度参数: %.2f%n%n", TEMPERATURE);

        for (int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++) {
            // 初始化 KV Cache
            List<List<Value[]>> keys = model.initKVCache();
            List<List<Value[]>> values = model.initKVCache();

            // 从 BOS 开始生成
            int tokenId = tokenizer.getBOS();
            StringBuilder output = new StringBuilder();

            for (int posId = 0; posId < BLOCK_SIZE; posId++) {
                // 前向传播
                Value[] logits = model.forward(tokenId, posId, keys, values);

                // 应用温度缩放
                Value[] scaledLogits = new Value[logits.length];
                for (int i = 0; i < logits.length; i++) {
                    scaledLogits[i] = logits[i].div(TEMPERATURE);
                }

                // 计算 softmax 概率
                Value[] probs = model.softmax(scaledLogits);

                // 根据概率分布采样下一个 token
                tokenId = sampleFromProbs(probs);

                // 如果遇到 BOS，停止生成
                if (tokenId == tokenizer.getBOS()) {
                    break;
                }

                // 解码并添加到输出
                output.append(tokenizer.decode(tokenId));
            }

            System.out.printf("sample %2d: %s%n", sampleIdx + 1, output.toString());
        }
    }

    /**
     * 根据概率分布采样
     *
     * @param probs 概率分布
     * @return 采样得到的索引
     */
    private int sampleFromProbs(Value[] probs) {
        // 将概率转换为 double 数组
        double[] p = new double[probs.length];
        for (int i = 0; i < probs.length; i++) {
            p[i] = probs[i].data;
        }

        // 确保概率和为 1（数值稳定性）
        double sum = 0;
        for (double v : p) {
            sum += v;
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= sum;
        }

        // 使用累积分布采样
        double r = random.nextDouble();
        double cumSum = 0;
        for (int i = 0; i < p.length; i++) {
            cumSum += p[i];
            if (r < cumSum) {
                return i;
            }
        }

        return p.length - 1;
    }

    /**
     * 生成指定数量的样本并打印
     *
     * @param numSamples 生成样本数量
     */
    private void generateSamples(int numSamples) {
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
            List<List<Value[]>> keys = model.initKVCache();
            List<List<Value[]>> values = model.initKVCache();

            int tokenId = tokenizer.getBOS();
            StringBuilder output = new StringBuilder();

            for (int posId = 0; posId < BLOCK_SIZE; posId++) {
                Value[] logits = model.forward(tokenId, posId, keys, values);

                Value[] scaledLogits = new Value[logits.length];
                for (int i = 0; i < logits.length; i++) {
                    scaledLogits[i] = logits[i].div(TEMPERATURE);
                }

                Value[] probs = model.softmax(scaledLogits);
                tokenId = sampleFromProbs(probs);

                if (tokenId == tokenizer.getBOS()) {
                    break;
                }
                output.append(tokenizer.decode(tokenId));
            }

            System.out.printf("sample %2d: %s%n", sampleIdx + 1, output.toString());
        }
    }

    /**
     * 计算模型的参数数量
     */
    private int countParameters() {
        return model.getParams().size();
    }

    /**
     * 打印模型配置
     */
    private void printConfig() {
        System.out.println("\n--- 模型配置 ---");
        System.out.println("嵌入维度: " + N_EMBD);
        System.out.println("注意力头数: " + N_HEAD);
        System.out.println("Transformer 层数: " + N_LAYER);
        System.out.println("最大序列长度: " + BLOCK_SIZE);
        System.out.println("参数数量: " + countParameters());
    }
}
