package io.leavesfly.microgpt;

import java.util.List;

/**
 * AdamOptimizer 类 - Adam 优化器实现
 * 
 * Adam (Adaptive Moment Estimation) 是一种自适应学习率优化算法。
 * 结合了 Momentum（动量）和 RMSprop（均方根传播）的优点。
 */
public class AdamOptimizer {
    /** 学习率 */
    private double learningRate;
    
    /** 一阶矩估计的指数衰减率 */
    private double beta1;
    
    /** 二阶矩估计的指数衰减率 */
    private double beta2;
    
    /** 数值稳定性常数 */
    private double eps;
    
    /** 一阶矩估计（动量）缓冲区 */
    private double[] m;
    
    /** 二阶矩估计缓冲区 */
    private double[] v;
    
    /** 参数数量 */
    private int paramCount;
    
    /**
     * 构造函数
     * @param learningRate 学习率
     * @param beta1 一阶矩衰减率（默认 0.9）
     * @param beta2 二阶矩衰减率（默认 0.95）
     * @param eps 数值稳定性常数（默认 1e-8）
     * @param paramCount 参数数量
     */
    public AdamOptimizer(double learningRate, double beta1, double beta2, double eps, int paramCount) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.paramCount = paramCount;
        
        // 初始化矩估计缓冲区
        this.m = new double[paramCount];
        this.v = new double[paramCount];
    }
    
    /**
     * 使用默认参数的构造函数
     * @param learningRate 学习率
     * @param paramCount 参数数量
     */
    public AdamOptimizer(double learningRate, int paramCount) {
        this(learningRate, 0.9, 0.95, 1e-8, paramCount);
    }
    
    /**
     * 执行一步优化更新
     * 更新所有参数的值
     * 
     * @param params 模型参数列表
     * @param currentStep 当前训练步数
     * @param totalSteps 总训练步数（用于学习率衰减）
     */
    public void step(List<Value> params, int currentStep, int totalSteps) {
        // 计算当前学习率（余弦衰减，最低衰减到初始学习率的 10%）
        double minLrRatio = 0.1;
        double cosineDecay = 0.5 * (1.0 + Math.cos(Math.PI * currentStep / totalSteps));
        double lrT = learningRate * (minLrRatio + (1.0 - minLrRatio) * cosineDecay);
        
        // 更新每个参数
        for (int i = 0; i < params.size(); i++) {
            Value p = params.get(i);
            double grad = p.grad;
            
            // 更新一阶矩估计（动量）
            // m = beta1 * m + (1 - beta1) * grad
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            
            // 更新二阶矩估计
            // v = beta2 * v + (1 - beta2) * grad^2
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            
            // 计算偏差修正后的一阶矩估计
            // m_hat = m / (1 - beta1^t)
            double mHat = m[i] / (1 - Math.pow(beta1, currentStep + 1));
            
            // 计算偏差修正后的二阶矩估计
            // v_hat = v / (1 - beta2^t)
            double vHat = v[i] / (1 - Math.pow(beta2, currentStep + 1));
            
            // 更新参数
            // p = p - lr * m_hat / (sqrt(v_hat) + eps)
            p.data -= lrT * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    /**
     * 执行一步优化更新（不使用学习率衰减）
     *
     * @param params      模型参数列表
     * @param currentStep 当前训练步数
     */
    public void step(List<Value> params, int currentStep) {
        step(params, currentStep, Integer.MAX_VALUE);
    }

    /**
     * 清零所有参数的梯度
     *
     * @param params 模型参数列表
     */
    public static void zeroGrad(List<Value> params) {
        for (Value p : params) {
            p.grad = 0;
        }
    }

    /**
     * 重置优化器状态
     * 清空所有矩估计缓冲区
     */
    public void reset() {
        this.m = new double[paramCount];
        this.v = new double[paramCount];
    }
    
    // ============ Getter 和 Setter ============
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public double getBeta1() {
        return beta1;
    }
    
    public double getBeta2() {
        return beta2;
    }
    
    public double getEps() {
        return eps;
    }
}
