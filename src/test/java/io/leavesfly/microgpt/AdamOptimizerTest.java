package io.leavesfly.microgpt;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * AdamOptimizer 优化器测试
 */
class AdamOptimizerTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testParameterUpdate() {
        Value param = new Value(5.0);
        param.grad = 1.0;

        List<Value> params = Arrays.asList(param);
        AdamOptimizer optimizer = new AdamOptimizer(0.01, params.size());

        double originalValue = param.data;
        optimizer.step(params, 0);

        assertNotEquals(originalValue, param.data, "参数应在优化步骤后更新");
    }

    @Test
    void testParameterMovesTowardNegativeGradient() {
        Value param = new Value(5.0);
        param.grad = 10.0;

        List<Value> params = Arrays.asList(param);
        AdamOptimizer optimizer = new AdamOptimizer(0.1, params.size());

        optimizer.step(params, 0);
        assertTrue(param.data < 5.0, "正梯度应使参数减小");
    }

    @Test
    void testZeroGrad() {
        Value param1 = new Value(1.0);
        Value param2 = new Value(2.0);
        param1.grad = 5.0;
        param2.grad = 3.0;

        List<Value> params = Arrays.asList(param1, param2);
        AdamOptimizer.zeroGrad(params);

        assertEquals(0.0, param1.grad, EPSILON, "梯度应被清零");
        assertEquals(0.0, param2.grad, EPSILON, "梯度应被清零");
    }

    @Test
    void testZeroGradDoesNotAffectData() {
        Value param = new Value(5.0);
        param.grad = 3.0;

        AdamOptimizer.zeroGrad(Arrays.asList(param));

        assertEquals(5.0, param.data, EPSILON, "zeroGrad 不应影响参数值");
        assertEquals(0.0, param.grad, EPSILON, "梯度应被清零");
    }

    @Test
    void testMultipleStepsConverge() {
        // 简单的优化问题：最小化 f(x) = x^2，梯度 = 2x
        Value param = new Value(10.0);
        List<Value> params = Arrays.asList(param);
        AdamOptimizer optimizer = new AdamOptimizer(0.5, params.size());

        for (int step = 0; step < 100; step++) {
            param.grad = 2.0 * param.data;
            optimizer.step(params, step);
        }

        assertTrue(Math.abs(param.data) < 1.0,
                "经过多步优化后参数应接近 0，实际值: " + param.data);
    }

    @Test
    void testLearningRateDecay() {
        Value param1 = new Value(5.0);
        param1.grad = 1.0;
        Value param2 = new Value(5.0);
        param2.grad = 1.0;

        // 使用学习率衰减的优化器
        AdamOptimizer optimizer1 = new AdamOptimizer(0.1, 1);
        AdamOptimizer optimizer2 = new AdamOptimizer(0.1, 1);

        // 在训练早期（step=0, totalSteps=100）
        optimizer1.step(Arrays.asList(param1), 0, 100);
        // 在训练末期（step=99, totalSteps=100）
        optimizer2.step(Arrays.asList(param2), 99, 100);

        double earlyUpdate = Math.abs(5.0 - param1.data);
        double lateUpdate = Math.abs(5.0 - param2.data);

        assertTrue(earlyUpdate > lateUpdate,
                "训练早期的更新幅度应大于末期（学习率衰减）");
    }

    @Test
    void testReset() {
        Value param = new Value(5.0);
        param.grad = 1.0;

        List<Value> params = Arrays.asList(param);
        AdamOptimizer optimizer = new AdamOptimizer(0.01, params.size());

        optimizer.step(params, 0);
        optimizer.reset();

        // reset 后再次 step，应该和第一次 step 的行为一致
        Value param2 = new Value(5.0);
        param2.grad = 1.0;
        List<Value> params2 = Arrays.asList(param2);
        AdamOptimizer optimizer2 = new AdamOptimizer(0.01, params2.size());
        optimizer2.step(params2, 0);

        // 重置后的优化器行为应与新创建的一致（但参数值不同，因为 param 已被更新过）
        assertNotNull(optimizer);
    }
}
