package io.leavesfly.microgpt;

import java.util.*;

/**
 * RewardFunction 类 - PPO 后训练的奖励函数
 *
 * 基于规则的奖励函数，用于评估生成的英文名字的质量。
 * 奖励信号引导模型生成更符合人类偏好的名字。
 */
public class RewardFunction {

    /** 训练集中的名字集合，用于新颖性检测 */
    private final Set<String> trainingNames;

    /** 元音字符集合 */
    private static final Set<Character> VOWELS = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u'));

    /** 常见英文名字首字母 */
    private static final Set<Character> COMMON_INITIALS = new HashSet<>(
            Arrays.asList('a', 'b', 'c', 'd', 'e', 'j', 'k', 'l', 'm', 'n', 'r', 's', 't'));

    /**
     * 构造函数
     *
     * @param trainingDocs 训练数据集，用于新颖性检测
     */
    public RewardFunction(List<String> trainingDocs) {
        this.trainingNames = new HashSet<>();
        for (String doc : trainingDocs) {
            this.trainingNames.add(doc.toLowerCase());
        }
    }

    /**
     * 计算生成序列的综合奖励分数
     *
     * @param generatedName 生成的名字字符串
     * @return 奖励分数（范围大约在 -2.0 到 4.0 之间）
     */
    public double score(String generatedName) {
        if (generatedName == null || generatedName.isEmpty()) {
            return -2.0;
        }

        double reward = 0.0;

        reward += lengthReward(generatedName);
        reward += vowelReward(generatedName);
        reward += initialReward(generatedName);
        reward += repetitionPenalty(generatedName);
        reward += noveltyReward(generatedName);
        reward += consonantVowelPatternReward(generatedName);

        return reward;
    }

    /**
     * 长度奖励：名字长度在 3-8 之间得正分，过短或过长扣分
     */
    private double lengthReward(String name) {
        int length = name.length();
        if (length >= 3 && length <= 8) {
            return 1.0;
        } else if (length == 2 || length == 9) {
            return 0.3;
        } else if (length == 1) {
            return -0.5;
        } else {
            return -1.0;
        }
    }

    /**
     * 元音奖励：名字中包含合理比例的元音字母
     * 英文名字通常元音占比在 30%-60% 之间
     */
    private double vowelReward(String name) {
        String lower = name.toLowerCase();
        int vowelCount = 0;
        for (char c : lower.toCharArray()) {
            if (VOWELS.contains(c)) {
                vowelCount++;
            }
        }

        double vowelRatio = (double) vowelCount / lower.length();
        if (vowelRatio >= 0.3 && vowelRatio <= 0.6) {
            return 0.8;
        } else if (vowelRatio > 0.0 && vowelRatio < 0.3) {
            return 0.2;
        } else if (vowelRatio > 0.6) {
            return 0.0;
        } else {
            return -0.5;
        }
    }

    /**
     * 首字母奖励：首字母是常见英文名字首字母
     */
    private double initialReward(String name) {
        char firstChar = Character.toLowerCase(name.charAt(0));
        if (COMMON_INITIALS.contains(firstChar)) {
            return 0.3;
        }
        return 0.0;
    }

    /**
     * 重复惩罚：连续 3 个以上相同字符扣分
     */
    private double repetitionPenalty(String name) {
        String lower = name.toLowerCase();
        int maxRepeat = 1;
        int currentRepeat = 1;

        for (int i = 1; i < lower.length(); i++) {
            if (lower.charAt(i) == lower.charAt(i - 1)) {
                currentRepeat++;
                maxRepeat = Math.max(maxRepeat, currentRepeat);
            } else {
                currentRepeat = 1;
            }
        }

        if (maxRepeat >= 3) {
            return -1.0;
        } else if (maxRepeat == 2) {
            return -0.2;
        }
        return 0.3;
    }

    /**
     * 新颖性奖励：生成的名字不在训练集中出现过
     */
    private double noveltyReward(String name) {
        if (trainingNames.contains(name.toLowerCase())) {
            return 0.0;
        }
        return 0.8;
    }

    /**
     * 辅音-元音交替模式奖励：
     * 自然的英文名字通常有辅音和元音的交替模式（如 "ma-ri-a"）
     */
    private double consonantVowelPatternReward(String name) {
        String lower = name.toLowerCase();
        int transitions = 0;

        for (int i = 1; i < lower.length(); i++) {
            boolean prevIsVowel = VOWELS.contains(lower.charAt(i - 1));
            boolean currIsVowel = VOWELS.contains(lower.charAt(i));
            if (prevIsVowel != currIsVowel) {
                transitions++;
            }
        }

        double transitionRatio = (double) transitions / Math.max(1, lower.length() - 1);
        if (transitionRatio >= 0.5) {
            return 0.8;
        } else if (transitionRatio >= 0.3) {
            return 0.4;
        }
        return 0.0;
    }
}
