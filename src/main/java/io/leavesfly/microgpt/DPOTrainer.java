package io.leavesfly.microgpt;

import java.util.*;

/**
 * DPOTrainer 类 - Direct Preference Optimization 训练器
 *
 * DPO 核心思想：
 * 1. 直接使用人类偏好数据优化语言模型，无需训练奖励模型
 * 2. 通过偏好对 (chosen vs rejected) 直接优化策略
 * 3. 使用 KL 散度约束防止模型偏离参考模型太远
 *
 * 损失函数：
 * L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]
 *
 * 其中：
 * - y_w: chosen response (好的回答)
 * - y_l: rejected response (差的回答)
 * - π_θ: 当前策略模型
 * - π_ref: 参考模型（冻结）
 * - β: KL 惩罚系数
 */
public class DPOTrainer {

    // ============ DPO 超参数 ============

    /** DPO 训练轮数 */
    private final int epochs;

    /** 批次大小 */
    private final int batchSize;

    /** KL 惩罚系数 β */
    private final double beta;

    /** 学习率 */
    private final double learningRate;

    /** 最大序列长度 */
    private final int maxSeqLength;

    // ============ 核心组件 ============

    /** 策略模型（被优化的模型） */
    private final GPT policyModel;

    /** 参考模型（冻结的，用于 KL 约束） */
    private final GPT referenceModel;

    /** 分词器 */
    private final Tokenizer tokenizer;

    /** Adam 优化器 */
    private final AdamOptimizer optimizer;

    /** 随机数生成器 */
    private final Random random;

    /**
     * 偏好数据对
     */
    public static class PreferencePair {
        /** 提示词/上下文（可选，本项目中名字生成无显式 prompt） */
        public String prompt;

        /** 被选中的回答（好的回答） */
        public String chosen;

        /** 被拒绝的回答（差的回答） */
        public String rejected;

        /** 缓存：参考模型对 chosen 的 log probability */
        public Double cachedRefLogProbChosen = null;

        /** 缓存：参考模型对 rejected 的 log probability */
        public Double cachedRefLogProbRejected = null;

        public PreferencePair(String chosen, String rejected) {
            this.prompt = "";
            this.chosen = chosen;
            this.rejected = rejected;
        }

        public PreferencePair(String prompt, String chosen, String rejected) {
            this.prompt = prompt;
            this.chosen = chosen;
            this.rejected = rejected;
        }
    }

    /**
     * DPO 超参数配置
     */
    public static class Config {
        /** DPO 训练轮数 */
        public int epochs = 100;

        /** 批次大小 */
        public int batchSize = 4;

        /** KL 惩罚系数 β */
        public double beta = 0.1;

        /** 学习率 */
        public double learningRate = 1e-4;

        public Config() {}
    }

    /**
     * 构造函数
     *
     * @param policyModel 策略模型（预训练后的模型）
     * @param tokenizer   分词器
     * @param config      DPO 超参数配置
     */
    public DPOTrainer(GPT policyModel, Tokenizer tokenizer, Config config) {
        this.policyModel = policyModel;
        this.referenceModel = policyModel.cloneModel();
        this.tokenizer = tokenizer;
        this.random = new Random(42);

        // 从配置中读取超参数
        this.epochs = config.epochs;
        this.batchSize = config.batchSize;
        this.beta = config.beta;
        this.learningRate = config.learningRate;
        this.maxSeqLength = policyModel.getBlockSize();

        this.optimizer = new AdamOptimizer(
                learningRate,
                0.9,
                0.95,
                1e-8,
                policyModel.getParams().size()
        );
    }

    /**
     * 使用默认配置的构造函数
     *
     * @param policyModel 策略模型（预训练后的模型）
     * @param tokenizer   分词器
     */
    public DPOTrainer(GPT policyModel, Tokenizer tokenizer) {
        this(policyModel, tokenizer, new Config());
    }

    /**
     * 执行 DPO 训练
     *
     * @param preferencePairs 偏好数据对列表
     */
    public void train(List<PreferencePair> preferencePairs) {
        System.out.println("\n--- DPO 后训练 ---");
        System.out.printf("Epochs: %d, Batch size: %d, β: %.3f, 学习率: %.5f%n",
                epochs, batchSize, beta, learningRate);
        System.out.println("偏好数据对数量: " + preferencePairs.size());

        if (preferencePairs.isEmpty()) {
            System.out.println("警告：没有偏好数据，跳过 DPO 训练");
            return;
        }

        long startTime = System.currentTimeMillis();

        // 预计算并缓存参考模型的 log probability（只需计算一次）
        System.out.println("预计算参考模型的 log probability...");
        for (PreferencePair pair : preferencePairs) {
            pair.cachedRefLogProbChosen = computeSequenceLogProbDetached(pair.chosen, referenceModel);
            pair.cachedRefLogProbRejected = computeSequenceLogProbDetached(pair.rejected, referenceModel);
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            // 打乱数据
            List<PreferencePair> shuffled = new ArrayList<>(preferencePairs);
            Collections.shuffle(shuffled, random);

            double epochLoss = 0;
            int numBatches = 0;

            // 分批训练
            for (int batchStart = 0; batchStart < shuffled.size(); batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, shuffled.size());
                List<PreferencePair> batch = shuffled.subList(batchStart, batchEnd);

                double batchLoss = trainBatch(batch);
                epochLoss += batchLoss;
                numBatches++;
            }

            double avgLoss = epochLoss / numBatches;

            // 打印进度
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                System.out.printf("epoch %4d / %4d | avg_loss %.4f%n", epoch + 1, epochs, avgLoss);
            }
        }

        long endTime = System.currentTimeMillis();
        System.out.printf("%nDPO 训练完成！(耗时: %.1fs)%n", (endTime - startTime) / 1000.0);
    }

    /**
     * 训练一个批次
     *
     * @param batch 偏好数据批次
     * @return 平均损失值
     */
    private double trainBatch(List<PreferencePair> batch) {
        // 清零梯度
        AdamOptimizer.zeroGrad(policyModel.getParams());

        Value totalLoss = new Value(0);

        for (PreferencePair pair : batch) {
            Value pairLoss = computeDPOLoss(pair);
            totalLoss = totalLoss.add(pairLoss);
        }

        // 取平均
        Value avgLoss = totalLoss.div(batch.size());

        // 反向传播
        avgLoss.backward();

        // 优化器更新
        optimizer.step(policyModel.getParams(), 0);

        return avgLoss.data;
    }

    /**
     * 计算单个偏好对的 DPO 损失
     *
     * L_DPO = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
     * 其中 log_ratio = log π_θ(y|x) - log π_ref(y|x)
     *
     * @param pair 偏好数据对
     * @return DPO 损失值
     */
    private Value computeDPOLoss(PreferencePair pair) {
        // 计算策略模型下 chosen 和 rejected 的 log probability
        Value logProbChosenPolicy = computeSequenceLogProb(pair.chosen, policyModel);
        Value logProbRejectedPolicy = computeSequenceLogProb(pair.rejected, policyModel);

        // 使用缓存的参考模型 log probability
        double logProbChosenRef = pair.cachedRefLogProbChosen;
        double logProbRejectedRef = pair.cachedRefLogProbRejected;

        // 计算 log ratio
        // log π_θ(y_w|x) - log π_ref(y_w|x)
        Value logRatioChosen = logProbChosenPolicy.sub(logProbChosenRef);
        // log π_θ(y_l|x) - log π_ref(y_l|x)
        Value logRatioRejected = logProbRejectedPolicy.sub(logProbRejectedRef);

        // 计算 logits: β * (log_ratio_chosen - log_ratio_rejected)
        Value logits = logRatioChosen.sub(logRatioRejected).mul(beta);

        // DPO Loss = -log σ(logits)
        // 使用 log(sigmoid(x)) = -log(1 + exp(-x)) 的数值稳定形式
        Value loss = logits.sigmoid().log().neg();

        return loss;
    }

    /**
     * 计算序列在模型下的 log probability（带梯度，用于策略模型）
     *
     * @param sequence 输入序列
     * @param model    模型
     * @return log P(sequence | model)
     */
    private Value computeSequenceLogProb(String sequence, GPT model) {
        int[] tokens = tokenizer.encode(sequence);

        List<List<Value[]>> keys = model.initKVCache();
        List<List<Value[]>> values = model.initKVCache();

        Value logProb = new Value(0);

        // 使用 BOS 作为起始 token
        int prevToken = tokenizer.getBOS();

        for (int i = 0; i < tokens.length && i < maxSeqLength; i++) {
            int currentToken = tokens[i];

            // 前向传播获取 logits
            Value[] logits = model.forward(prevToken, i, keys, values);

            // 计算 softmax 概率
            Value[] probs = model.softmax(logits);

            // 获取当前 token 的概率并取 log
            Value tokenLogProb = probs[currentToken].log();
            logProb = logProb.add(tokenLogProb);

            prevToken = currentToken;
        }

        return logProb;
    }

    /**
     * 计算序列在模型下的 log probability（不带梯度，用于参考模型）
     *
     * @param sequence 输入序列
     * @param model    模型
     * @return log P(sequence | model)
     */
    private double computeSequenceLogProbDetached(String sequence, GPT model) {
        int[] tokens = tokenizer.encode(sequence);

        List<List<Value[]>> keys = model.initKVCache();
        List<List<Value[]>> values = model.initKVCache();

        double logProb = 0.0;

        int prevToken = tokenizer.getBOS();

        for (int i = 0; i < tokens.length && i < maxSeqLength; i++) {
            int currentToken = tokens[i];

            Value[] logits = model.forward(prevToken, i, keys, values);
            Value[] probs = model.softmax(logits);

            // 只取数值，不保留计算图
            logProb += Math.log(Math.max(probs[currentToken].data, 1e-10));

            prevToken = currentToken;
        }

        return logProb;
    }

    /**
     * 使用 DPO 训练后的模型生成样本
     *
     * @param numSamples  生成样本数量
     * @param temperature 温度参数
     */
    public void generateSamples(int numSamples, double temperature) {
        System.out.println("\n--- DPO 后训练推理 ---");
        System.out.printf("温度参数: %.2f%n%n", temperature);

        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
            List<List<Value[]>> keys = policyModel.initKVCache();
            List<List<Value[]>> values = policyModel.initKVCache();

            int tokenId = tokenizer.getBOS();
            StringBuilder output = new StringBuilder();

            for (int posId = 0; posId < maxSeqLength; posId++) {
                Value[] logits = policyModel.forward(tokenId, posId, keys, values);

                Value[] scaledLogits = new Value[logits.length];
                for (int i = 0; i < logits.length; i++) {
                    scaledLogits[i] = logits[i].div(temperature);
                }

                Value[] probs = policyModel.softmax(scaledLogits);
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
     * 根据概率分布采样
     *
     * @param probs 概率分布
     * @return 采样得到的 token 索引
     */
    private int sampleFromProbs(Value[] probs) {
        double[] probArray = new double[probs.length];
        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            probArray[i] = probs[i].data;
            sum += probArray[i];
        }
        for (int i = 0; i < probArray.length; i++) {
            probArray[i] /= sum;
        }

        double r = random.nextDouble();
        double cumSum = 0;
        for (int i = 0; i < probArray.length; i++) {
            cumSum += probArray[i];
            if (r < cumSum) {
                return i;
            }
        }
        return probArray.length - 1;
    }

    /**
     * 获取策略模型
     */
    public GPT getPolicyModel() {
        return policyModel;
    }
}
