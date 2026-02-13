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
    private static final int NUM_STEPS = 200;

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

    // ============ PPO 后训练超参数 ============

    /**
     * PPO 训练步数
     */
    private static final int PPO_STEPS = 200;

    /**
     * 每步生成的 rollout 数量（越大梯度估计越准确）
     */
    private static final int PPO_ROLLOUTS_PER_STEP = 8;

    /**
     * PPO clip 范围 ε（控制每步更新幅度）
     */
    private static final double PPO_CLIP_EPSILON = 0.2;

    /**
     * KL 散度惩罚系数 β（越小允许策略探索越多）
     */
    private static final double PPO_KL_COEFFICIENT = 0.05;

    /**
     * PPO 学习率
     */
    private static final double PPO_LEARNING_RATE = 1e-3;

    /**
     * PPO 推理温度
     */
    private static final double PPO_TEMPERATURE = 0.8;

    /**
     * Reference model 同步间隔
     */
    private static final int PPO_REF_SYNC_INTERVAL = 25;

    // ============ DPO 后训练超参数 ============

    /**
     * DPO 训练轮数
     */
    private static final int DPO_EPOCHS = 100;

    /**
     * DPO 批次大小
     */
    private static final int DPO_BATCH_SIZE = 16;

    /**
     * DPO KL 惩罚系数 β
     */
    private static final double DPO_BETA = 0.1;

    /**
     * DPO 学习率
     */
    private static final double DPO_LEARNING_RATE = 5e-4;

    // ============ 核心组件 ============

    private Tokenizer tokenizer;
    private GPT model;
    private AdamOptimizer optimizer;
    private PPOTrainer ppoTrainer;
    private DPOTrainer dpoTrainer;
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

        // 7. 预训练后推理生成
        inference();

        // 8. PPO 后训练（强化学习）
        ppoPostTrain();

        // 9. PPO 后训练推理生成
        ppoInference();

        // 10. DPO 后训练（直接偏好优化）
        dpoPostTrain();

        // 11. DPO 后训练推理生成
        dpoInference();
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
     * PPO 后训练（强化学习阶段）
     * 在预训练完成后，使用 PPO 算法进一步优化模型
     */
    private void ppoPostTrain() {
        RewardFunction rewardFunction = new RewardFunction(docs);

        PPOTrainer.Config ppoConfig = new PPOTrainer.Config();
        ppoConfig.ppoSteps = PPO_STEPS;
        ppoConfig.rolloutsPerStep = PPO_ROLLOUTS_PER_STEP;
        ppoConfig.clipEpsilon = PPO_CLIP_EPSILON;
        ppoConfig.klCoefficient = PPO_KL_COEFFICIENT;
        ppoConfig.learningRate = PPO_LEARNING_RATE;
        ppoConfig.temperature = PPO_TEMPERATURE;
        ppoConfig.refSyncInterval = PPO_REF_SYNC_INTERVAL;

        ppoTrainer = new PPOTrainer(model, rewardFunction, tokenizer, ppoConfig);
        ppoTrainer.train();
    }

    /**
     * PPO 后训练推理生成
     * 展示 PPO 后训练后的模型生成效果
     */
    private void ppoInference() {
        ppoTrainer.generateSamples(NUM_SAMPLES, TEMPERATURE);
    }

    /**
     * DPO 后训练（直接偏好优化阶段）
     * 在预训练完成后，使用 DPO 算法进一步优化模型
     */
    private void dpoPostTrain() {
        // 构建偏好数据：从训练数据中生成偏好对
        List<DPOTrainer.PreferencePair> preferencePairs = generatePreferencePairs();

        DPOTrainer.Config dpoConfig = new DPOTrainer.Config();
        dpoConfig.epochs = DPO_EPOCHS;
        dpoConfig.batchSize = DPO_BATCH_SIZE;
        dpoConfig.beta = DPO_BETA;
        dpoConfig.learningRate = DPO_LEARNING_RATE;

        // 使用 PPO 训练后的模型进行 DPO
        dpoTrainer = new DPOTrainer(model, tokenizer, dpoConfig);
        dpoTrainer.train(preferencePairs);
    }

    /**
     * DPO 后训练推理生成
     * 展示 DPO 后训练后的模型生成效果
     */
    private void dpoInference() {
        dpoTrainer.generateSamples(NUM_SAMPLES, TEMPERATURE);
    }

    /**
     * 生成偏好数据对
     * 基于规则奖励函数生成 chosen vs rejected 偏好对
     */
    private List<DPOTrainer.PreferencePair> generatePreferencePairs() {
        List<DPOTrainer.PreferencePair> pairs = new ArrayList<>();
        RewardFunction rewardFunction = new RewardFunction(docs);

        // 使用当前模型生成样本，并根据奖励分数构造偏好对
        int numSamples = 100;  // 生成样本数量
        List<String> generated = new ArrayList<>();
        List<Double> rewards = new ArrayList<>();

        for (int i = 0; i < numSamples; i++) {
            List<List<Value[]>> keys = model.initKVCache();
            List<List<Value[]>> values = model.initKVCache();

            int tokenId = tokenizer.getBOS();
            StringBuilder output = new StringBuilder();

            for (int posId = 0; posId < BLOCK_SIZE; posId++) {
                Value[] logits = model.forward(tokenId, posId, keys, values);

                Value[] scaledLogits = new Value[logits.length];
                for (int j = 0; j < logits.length; j++) {
                    scaledLogits[j] = logits[j].div(TEMPERATURE);
                }

                Value[] probs = model.softmax(scaledLogits);
                tokenId = sampleFromProbs(probs);

                if (tokenId == tokenizer.getBOS()) {
                    break;
                }
                output.append(tokenizer.decode(tokenId));
            }

            String text = output.toString();
            double reward = rewardFunction.score(text);

            generated.add(text);
            rewards.add(reward);
        }

        // 根据奖励分数构建偏好对（只取高/低分对比，避免组合爆炸）
        // 按奖励排序
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < generated.size(); i++) indices.add(i);
        indices.sort((a, b) -> Double.compare(rewards.get(b), rewards.get(a)));

        // 高分 vs 低分配对（最多取 min(高分组, 低分组) 个对）
        int halfSize = indices.size() / 2;
        for (int i = 0; i < halfSize; i++) {
            int highIdx = indices.get(i);
            int lowIdx = indices.get(indices.size() - 1 - i);
            if (rewards.get(highIdx) > rewards.get(lowIdx)) {
                pairs.add(new DPOTrainer.PreferencePair(generated.get(highIdx), generated.get(lowIdx)));
            }
        }

        System.out.println("生成偏好数据对数量: " + pairs.size());
        return pairs;
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
