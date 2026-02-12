package io.leavesfly.microgpt;

import java.util.*;

/**
 * PPOTrainer 类 - Proximal Policy Optimization 训练器
 *
 * 实现 PPO-Clip 算法，用于 GPT 模型的后训练（RLHF 风格）。
 *
 * PPO 核心思想：
 * 1. 用当前策略（policy model）生成序列并收集 rollout 数据
 * 2. 计算奖励和优势估计（advantage）
 * 3. 使用 clipped surrogate objective 更新策略，防止更新幅度过大
 * 4. 加入 KL 散度惩罚，防止策略偏离参考模型（reference model）太远
 *
 * 损失函数：L = -min(rₜ·Aₜ, clip(rₜ, 1-ε, 1+ε)·Aₜ) + β·KL(π‖π_ref)
 */
public class PPOTrainer {

    // ============ PPO 超参数 ============

    /** PPO 训练步数 */
    private final int ppoSteps;

    /** 每步生成的 rollout 数量 */
    private final int rolloutsPerStep;

    /** PPO clip 范围 ε */
    private final double clipEpsilon;

    /** KL 散度惩罚系数 β */
    private final double klCoefficient;

    /** PPO 学习率 */
    private final double learningRate;

    /** 推理温度 */
    private final double temperature;

    /** 最大生成长度 */
    private final int maxGenerateLength;

    /** GAE 折扣因子 γ */
    private final double gamma;

    /** Reference model 同步间隔（每隔多少步同步一次） */
    private final int refSyncInterval;

    // ============ 核心组件 ============

    /** 策略模型（被优化的模型） */
    private final GPT policyModel;

    /** 参考模型（冻结的，用于 KL 约束） */
    private final GPT referenceModel;

    /** 奖励函数 */
    private final RewardFunction rewardFunction;

    /** 分词器 */
    private final Tokenizer tokenizer;

    /** Adam 优化器 */
    private final AdamOptimizer optimizer;

    /** 随机数生成器 */
    private final Random random;

    /**
     * Rollout 数据：记录一次完整生成过程中的所有信息
     */
    private static class RolloutData {
        /** 生成的 token 序列 */
        List<Integer> generatedTokens = new ArrayList<>();

        /** 每步的 log π_policy(aₜ|sₜ)（带计算图，用于反向传播） */
        List<Value> policyLogProbs = new ArrayList<>();

        /** 每步的 log π_ref(aₜ|sₜ)（detach，不参与梯度计算） */
        List<Double> referenceLogProbs = new ArrayList<>();

        /** 生成的文本 */
        String generatedText = "";

        /** 奖励分数 */
        double reward = 0.0;
    }

    /**
     * PPO 超参数配置
     */
    public static class Config {
        /** PPO 训练步数 */
        public int ppoSteps = 200;

        /** 每步生成的 rollout 数量 */
        public int rolloutsPerStep = 8;

        /** PPO clip 范围 ε */
        public double clipEpsilon = 0.2;

        /** KL 散度惩罚系数 β */
        public double klCoefficient = 0.05;

        /** PPO 学习率 */
        public double learningRate = 1e-3;

        /** 推理温度 */
        public double temperature = 0.8;

        /** GAE 折扣因子 γ */
        public double gamma = 1.0;

        /** Reference model 同步间隔（每隔多少步同步一次） */
        public int refSyncInterval = 25;
    }

    /**
     * 构造函数
     *
     * @param policyModel    策略模型（预训练后的模型）
     * @param rewardFunction 奖励函数
     * @param tokenizer      分词器
     * @param config         PPO 超参数配置
     */
    public PPOTrainer(GPT policyModel, RewardFunction rewardFunction, Tokenizer tokenizer, Config config) {
        this.policyModel = policyModel;
        this.referenceModel = policyModel.cloneModel();
        this.rewardFunction = rewardFunction;
        this.tokenizer = tokenizer;
        this.random = new Random(42);

        // 从配置中读取超参数
        this.ppoSteps = config.ppoSteps;
        this.rolloutsPerStep = config.rolloutsPerStep;
        this.clipEpsilon = config.clipEpsilon;
        this.klCoefficient = config.klCoefficient;
        this.learningRate = config.learningRate;
        this.temperature = config.temperature;
        this.maxGenerateLength = policyModel.getBlockSize();
        this.gamma = config.gamma;
        this.refSyncInterval = config.refSyncInterval;

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
     * @param policyModel    策略模型（预训练后的模型）
     * @param rewardFunction 奖励函数
     * @param tokenizer      分词器
     */
    public PPOTrainer(GPT policyModel, RewardFunction rewardFunction, Tokenizer tokenizer) {
        this(policyModel, rewardFunction, tokenizer, new Config());
    }

    /**
     * 执行 PPO 后训练
     */
    public void train() {
        System.out.println("\n--- PPO 后训练 ---");
        System.out.printf("PPO 步数: %d, Rollouts/步: %d, Clip ε: %.2f, KL β: %.2f, 学习率: %.4f%n%n",
                ppoSteps, rolloutsPerStep, clipEpsilon, klCoefficient, learningRate);

        long startTime = System.currentTimeMillis();
        double smoothReward = Double.NaN;

        for (int step = 0; step < ppoSteps; step++) {
            // ============ Phase 1: 收集 Rollout 数据 ============
            List<RolloutData> rollouts = collectRollouts();

            // 计算本步的平均奖励
            double avgReward = 0;
            for (RolloutData rollout : rollouts) {
                avgReward += rollout.reward;
            }
            avgReward /= rollouts.size();
            smoothReward = Double.isNaN(smoothReward) ? avgReward : 0.9 * smoothReward + 0.1 * avgReward;

            // ============ Phase 2: 计算优势估计 ============
            double[] advantages = computeAdvantages(rollouts);

            // ============ Phase 3: PPO 策略更新 ============
            double ppoLoss = ppoUpdate(rollouts, advantages, step);

            // 打印进度
            if ((step + 1) % 10 == 0 || step == 0) {
                System.out.printf("ppo_step %4d / %4d | avg_reward %.4f | smooth_reward %.4f | ppo_loss %.4f%n",
                        step + 1, ppoSteps, avgReward, smoothReward, ppoLoss);
            }

            // 定期同步 reference model
            if ((step + 1) % refSyncInterval == 0) {
                referenceModel.syncParamsFrom(policyModel);
            }
        }

        long endTime = System.currentTimeMillis();
        System.out.printf("%nPPO 后训练完成！(耗时: %.1fs)%n", (endTime - startTime) / 1000.0);
    }

    /**
     * 收集一批 Rollout 数据
     * 使用当前策略模型和参考模型分别生成序列，记录 log 概率
     */
    private List<RolloutData> collectRollouts() {
        List<RolloutData> rollouts = new ArrayList<>();

        for (int r = 0; r < rolloutsPerStep; r++) {
            RolloutData rollout = new RolloutData();

            // 初始化 KV Cache（策略模型和参考模型各自独立）
            List<List<Value[]>> policyKeys = policyModel.initKVCache();
            List<List<Value[]>> policyValues = policyModel.initKVCache();
            List<List<Value[]>> refKeys = referenceModel.initKVCache();
            List<List<Value[]>> refValues = referenceModel.initKVCache();

            int tokenId = tokenizer.getBOS();
            StringBuilder output = new StringBuilder();

            for (int posId = 0; posId < maxGenerateLength; posId++) {
                // --- 策略模型前向传播 ---
                Value[] policyLogits = policyModel.forward(tokenId, posId, policyKeys, policyValues);

                // 应用温度缩放
                Value[] scaledPolicyLogits = new Value[policyLogits.length];
                for (int i = 0; i < policyLogits.length; i++) {
                    scaledPolicyLogits[i] = policyLogits[i].div(temperature);
                }
                Value[] policyProbs = policyModel.softmax(scaledPolicyLogits);

                // --- 参考模型前向传播（不参与梯度计算） ---
                Value[] refLogits = referenceModel.forward(tokenId, posId, refKeys, refValues);
                Value[] scaledRefLogits = new Value[refLogits.length];
                for (int i = 0; i < refLogits.length; i++) {
                    scaledRefLogits[i] = refLogits[i].div(temperature);
                }
                Value[] refProbs = referenceModel.softmax(scaledRefLogits);

                // 从策略模型的概率分布中采样
                tokenId = sampleFromProbs(policyProbs);

                // 如果遇到 BOS（序列结束），停止生成
                if (tokenId == tokenizer.getBOS()) {
                    break;
                }

                // 记录数据
                rollout.generatedTokens.add(tokenId);
                rollout.policyLogProbs.add(policyProbs[tokenId].log());
                rollout.referenceLogProbs.add(Math.log(Math.max(refProbs[tokenId].data, 1e-10)));

                output.append(tokenizer.decode(tokenId));
            }

            rollout.generatedText = output.toString();
            rollout.reward = rewardFunction.score(rollout.generatedText);
            rollouts.add(rollout);
        }

        return rollouts;
    }

    /**
     * 计算优势估计（Advantage Estimation）
     * 使用简化的优势计算：A = R - baseline
     * baseline 使用当前 batch 的平均奖励
     */
    private double[] computeAdvantages(List<RolloutData> rollouts) {
        // 计算 baseline（当前 batch 的平均奖励）
        double baseline = 0;
        for (RolloutData rollout : rollouts) {
            baseline += rollout.reward;
        }
        baseline /= rollouts.size();

        // 计算每个 rollout 的优势值
        double[] advantages = new double[rollouts.size()];
        for (int i = 0; i < rollouts.size(); i++) {
            advantages[i] = rollouts.get(i).reward - baseline;
        }

        // 优势标准化（减少方差）
        double mean = 0;
        for (double adv : advantages) {
            mean += adv;
        }
        mean /= advantages.length;

        double variance = 0;
        for (double adv : advantages) {
            variance += (adv - mean) * (adv - mean);
        }
        variance /= advantages.length;
        double std = Math.sqrt(variance + 1e-8);

        for (int i = 0; i < advantages.length; i++) {
            advantages[i] = (advantages[i] - mean) / std;
        }

        return advantages;
    }

    /**
     * PPO 策略更新
     * 使用 Clipped Surrogate Objective + KL 惩罚
     *
     * @param rollouts   Rollout 数据
     * @param advantages 优势估计
     * @param step       当前训练步数
     * @return PPO 损失值
     */
    private double ppoUpdate(List<RolloutData> rollouts, double[] advantages, int step) {
        // 清零梯度
        AdamOptimizer.zeroGrad(policyModel.getParams());

        double totalLoss = 0;
        int totalTokens = 0;

        for (int r = 0; r < rollouts.size(); r++) {
            RolloutData rollout = rollouts.get(r);
            double advantage = advantages[r];

            if (rollout.policyLogProbs.isEmpty()) {
                continue;
            }

            // 重新用策略模型前向传播，获取当前策略下的 log 概率（带梯度）
            List<List<Value[]>> newKeys = policyModel.initKVCache();
            List<List<Value[]>> newValues = policyModel.initKVCache();

            int currentTokenId = tokenizer.getBOS();
            Value rolloutLoss = new Value(0);
            int tokenCount = 0;

            for (int t = 0; t < rollout.generatedTokens.size(); t++) {
                // 前向传播获取当前策略的 logits
                Value[] logits = policyModel.forward(currentTokenId, t, newKeys, newValues);
                Value[] scaledLogits = new Value[logits.length];
                for (int i = 0; i < logits.length; i++) {
                    scaledLogits[i] = logits[i].div(temperature);
                }
                Value[] probs = policyModel.softmax(scaledLogits);

                int actionToken = rollout.generatedTokens.get(t);
                Value newLogProb = probs[actionToken].log();

                // 旧的 log 概率（detach，不参与当前梯度计算）
                double oldLogProb = rollout.policyLogProbs.get(t).data;

                // 计算概率比 rₜ = π_new(aₜ|sₜ) / π_old(aₜ|sₜ) = exp(log_new - log_old)
                Value ratio = newLogProb.sub(oldLogProb).exp();

                // Clipped Surrogate Objective
                // L_clip = min(rₜ·Aₜ, clip(rₜ, 1-ε, 1+ε)·Aₜ)
                Value surrogate = ratio.mul(advantage);
                double clippedRatioVal = Math.max(1.0 - clipEpsilon, Math.min(1.0 + clipEpsilon, ratio.data));
                double clippedSurrogate = clippedRatioVal * advantage;

                // 取两者中较小的（保守更新）
                Value policyLoss;
                if (surrogate.data < clippedSurrogate) {
                    policyLoss = surrogate.mul(-1);
                } else {
                    // clipped 分支：用 ratio 的梯度乘以 clip 后的系数
                    policyLoss = surrogate.mul(-1);
                    // 当 ratio 超出 clip 范围时，梯度为 0（通过 detach 实现）
                    if (ratio.data < 1.0 - clipEpsilon || ratio.data > 1.0 + clipEpsilon) {
                        policyLoss = new Value(clippedSurrogate * -1);
                    }
                }

                // KL 散度惩罚：KL(π_new ‖ π_ref) ≈ log(π_new/π_ref) = log_π_new - log_π_ref
                double refLogProb = rollout.referenceLogProbs.get(t);
                Value klPenalty = newLogProb.sub(refLogProb).mul(klCoefficient);

                // 总损失 = 策略损失 + KL 惩罚
                Value stepLoss = policyLoss.add(klPenalty);
                rolloutLoss = rolloutLoss.add(stepLoss);
                tokenCount++;

                currentTokenId = actionToken;
            }

            if (tokenCount > 0) {
                // 对 token 数量和 rollout 数量取平均
                Value avgRolloutLoss = rolloutLoss.div(tokenCount * rollouts.size());
                avgRolloutLoss.backward();
                totalLoss += avgRolloutLoss.data * tokenCount * rollouts.size();
                totalTokens += tokenCount;
            }
        }

        // Adam 优化器更新参数
        optimizer.step(policyModel.getParams(), step, ppoSteps);

        return totalTokens > 0 ? totalLoss / totalTokens : 0;
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
     * 使用 PPO 训练后的模型生成样本
     *
     * @param numSamples  生成样本数量
     * @param temperature 温度参数
     */
    public void generateSamples(int numSamples, double temperature) {
        System.out.println("\n--- PPO 后训练推理 ---");
        System.out.printf("温度参数: %.2f%n%n", temperature);

        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
            List<List<Value[]>> keys = policyModel.initKVCache();
            List<List<Value[]>> values = policyModel.initKVCache();

            int tokenId = tokenizer.getBOS();
            StringBuilder output = new StringBuilder();

            for (int posId = 0; posId < maxGenerateLength; posId++) {
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

            String generated = output.toString();
            double reward = rewardFunction.score(generated);
            System.out.printf("sample %2d: %-12s (reward: %.2f)%n", sampleIdx + 1, generated, reward);
        }
    }
}
