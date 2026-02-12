package io.leavesfly.microgpt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Value 自动微分引擎测试
 */
class ValueTest {

    private static final double EPSILON = 1e-6;

    // ============ 前向计算测试 ============

    @Test
    void testAdd() {
        Value a = new Value(3.0);
        Value b = new Value(4.0);
        Value c = a.add(b);
        assertEquals(7.0, c.data, EPSILON);
    }

    @Test
    void testAddDouble() {
        Value a = new Value(3.0);
        Value c = a.add(5.0);
        assertEquals(8.0, c.data, EPSILON);
    }

    @Test
    void testMul() {
        Value a = new Value(3.0);
        Value b = new Value(4.0);
        Value c = a.mul(b);
        assertEquals(12.0, c.data, EPSILON);
    }

    @Test
    void testMulDouble() {
        Value a = new Value(3.0);
        Value c = a.mul(2.0);
        assertEquals(6.0, c.data, EPSILON);
    }

    @Test
    void testSub() {
        Value a = new Value(7.0);
        Value b = new Value(3.0);
        Value c = a.sub(b);
        assertEquals(4.0, c.data, EPSILON);
    }

    @Test
    void testSubDouble() {
        Value a = new Value(7.0);
        Value c = a.sub(3.0);
        assertEquals(4.0, c.data, EPSILON);
    }

    @Test
    void testDiv() {
        Value a = new Value(10.0);
        Value b = new Value(4.0);
        Value c = a.div(b);
        assertEquals(2.5, c.data, EPSILON);
    }

    @Test
    void testDivDouble() {
        Value a = new Value(10.0);
        Value c = a.div(4.0);
        assertEquals(2.5, c.data, EPSILON);
    }

    @Test
    void testPow() {
        Value a = new Value(3.0);
        Value c = a.pow(2.0);
        assertEquals(9.0, c.data, EPSILON);
    }

    @Test
    void testExp() {
        Value a = new Value(1.0);
        Value c = a.exp();
        assertEquals(Math.E, c.data, EPSILON);
    }

    @Test
    void testLog() {
        Value a = new Value(Math.E);
        Value c = a.log();
        assertEquals(1.0, c.data, EPSILON);
    }

    @Test
    void testRelu() {
        Value positive = new Value(3.0);
        Value negative = new Value(-3.0);
        assertEquals(3.0, positive.relu().data, EPSILON);
        assertEquals(0.0, negative.relu().data, EPSILON);
    }

    @Test
    void testNeg() {
        Value a = new Value(5.0);
        Value c = a.neg();
        assertEquals(-5.0, c.data, EPSILON);
    }

    // ============ 反向传播梯度测试 ============

    @Test
    void testAddBackward() {
        Value a = new Value(3.0);
        Value b = new Value(4.0);
        Value c = a.add(b);
        c.backward();
        assertEquals(1.0, a.grad, EPSILON, "d(a+b)/da = 1");
        assertEquals(1.0, b.grad, EPSILON, "d(a+b)/db = 1");
    }

    @Test
    void testMulBackward() {
        Value a = new Value(3.0);
        Value b = new Value(4.0);
        Value c = a.mul(b);
        c.backward();
        assertEquals(4.0, a.grad, EPSILON, "d(a*b)/da = b");
        assertEquals(3.0, b.grad, EPSILON, "d(a*b)/db = a");
    }

    @Test
    void testSubBackward() {
        Value a = new Value(7.0);
        Value b = new Value(3.0);
        Value c = a.sub(b);
        c.backward();
        assertEquals(1.0, a.grad, EPSILON, "d(a-b)/da = 1");
        assertEquals(-1.0, b.grad, EPSILON, "d(a-b)/db = -1");
    }

    @Test
    void testDivBackward() {
        Value a = new Value(10.0);
        Value b = new Value(4.0);
        Value c = a.div(b);
        c.backward();
        assertEquals(1.0 / 4.0, a.grad, EPSILON, "d(a/b)/da = 1/b");
        assertEquals(-10.0 / 16.0, b.grad, EPSILON, "d(a/b)/db = -a/b^2");
    }

    @Test
    void testDivDoubleBackward() {
        Value a = new Value(10.0);
        Value c = a.div(4.0);
        c.backward();
        assertEquals(1.0 / 4.0, a.grad, EPSILON, "d(a/4)/da = 1/4");
    }

    @Test
    void testPowBackward() {
        Value a = new Value(3.0);
        Value c = a.pow(2.0);
        c.backward();
        assertEquals(6.0, a.grad, EPSILON, "d(a^2)/da = 2a = 6");
    }

    @Test
    void testReluBackward() {
        Value positive = new Value(3.0);
        Value resultPositive = positive.relu();
        resultPositive.backward();
        assertEquals(1.0, positive.grad, EPSILON, "relu'(x>0) = 1");

        Value negative = new Value(-3.0);
        Value resultNegative = negative.relu();
        resultNegative.backward();
        assertEquals(0.0, negative.grad, EPSILON, "relu'(x<0) = 0");
    }

    @Test
    void testExpBackward() {
        Value a = new Value(2.0);
        Value c = a.exp();
        c.backward();
        assertEquals(Math.exp(2.0), a.grad, EPSILON, "d(exp(a))/da = exp(a)");
    }

    @Test
    void testLogBackward() {
        Value a = new Value(5.0);
        Value c = a.log();
        c.backward();
        assertEquals(1.0 / 5.0, a.grad, EPSILON, "d(log(a))/da = 1/a");
    }

    // ============ 复合表达式梯度测试 ============

    @Test
    void testCompositeExpression() {
        // f(a, b) = a * b + a^2
        // df/da = b + 2a = 4 + 6 = 10
        // df/db = a = 3
        Value a = new Value(3.0);
        Value b = new Value(4.0);
        Value c = a.mul(b).add(a.pow(2.0));
        c.backward();
        assertEquals(10.0, a.grad, EPSILON);
        assertEquals(3.0, b.grad, EPSILON);
    }

    @Test
    void testChainedOperations() {
        // f(x) = relu((x * 2 + 1) - 3) = relu(2x - 2)
        // x = 5 => f = relu(8) = 8
        // df/dx = 2
        Value x = new Value(5.0);
        Value result = x.mul(2.0).add(1.0).sub(3.0).relu();
        assertEquals(8.0, result.data, EPSILON);
        result.backward();
        assertEquals(2.0, x.grad, EPSILON);
    }

    // ============ 数值稳定性测试 ============

    @Test
    void testLogNumericalStability() {
        Value nearZero = new Value(0.0);
        Value result = nearZero.log();
        assertFalse(Double.isNaN(result.data), "log(0) should not produce NaN");
        assertFalse(Double.isInfinite(result.data), "log(0) should not produce Infinity");
    }

    @Test
    void testExpNumericalStability() {
        Value large = new Value(1000.0);
        Value result = large.exp();
        assertFalse(Double.isNaN(result.data), "exp(1000) should not produce NaN");
        assertFalse(Double.isInfinite(result.data), "exp(1000) should be clamped");
    }
}
