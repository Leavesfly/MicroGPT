package io.leavesfly.microgpt;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * GPT 模型测试
 */
class GPTTest {

    private static final double EPSILON = 1e-6;

    private GPT model;
    private static final int VOCAB_SIZE = 10;
    private static final int N_EMBD = 8;
    private static final int N_HEAD = 2;
    private static final int N_LAYER = 1;
    private static final int BLOCK_SIZE = 4;

    @BeforeEach
    void setUp() {
        model = new GPT(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE);
    }

    @Test
    void testModelInitialization() {
        assertNotNull(model.getParams());
        assertTrue(model.getParams().size() > 0, "模型应有参数");
        assertEquals(VOCAB_SIZE, model.getVocabSize());
        assertEquals(N_EMBD, model.getNEmbd());
        assertEquals(N_HEAD, model.getNHead());
        assertEquals(N_LAYER, model.getNLayer());
        assertEquals(BLOCK_SIZE, model.getBlockSize());
    }

    @Test
    void testForwardOutputDimension() {
        List<List<Value[]>> keys = model.initKVCache();
        List<List<Value[]>> values = model.initKVCache();

        Value[] logits = model.forward(0, 0, keys, values);
        assertEquals(VOCAB_SIZE, logits.length, "输出 logits 维度应等于词表大小");
    }

    @Test
    void testForwardProducesFiniteValues() {
        List<List<Value[]>> keys = model.initKVCache();
        List<List<Value[]>> values = model.initKVCache();

        Value[] logits = model.forward(1, 0, keys, values);
        for (int i = 0; i < logits.length; i++) {
            assertFalse(Double.isNaN(logits[i].data), "logits[" + i + "] 不应为 NaN");
            assertFalse(Double.isInfinite(logits[i].data), "logits[" + i + "] 不应为 Infinity");
        }
    }

    @Test
    void testMultipleForwardWithKVCache() {
        List<List<Value[]>> keys = model.initKVCache();
        List<List<Value[]>> values = model.initKVCache();

        // 模拟多步前向传播
        for (int posId = 0; posId < BLOCK_SIZE; posId++) {
            Value[] logits = model.forward(posId % VOCAB_SIZE, posId, keys, values);
            assertEquals(VOCAB_SIZE, logits.length);
        }

        // KV Cache 应该累积了 BLOCK_SIZE 个条目
        assertEquals(BLOCK_SIZE, keys.get(0).size(), "KV Cache 应累积正确数量的条目");
    }

    @Test
    void testSoftmaxProbabilitySumToOne() {
        Value[] logits = new Value[]{new Value(1.0), new Value(2.0), new Value(3.0)};
        Value[] probs = model.softmax(logits);

        double sum = 0;
        for (Value p : probs) {
            sum += p.data;
            assertTrue(p.data >= 0, "概率应非负");
            assertTrue(p.data <= 1, "概率应不超过 1");
        }
        assertEquals(1.0, sum, EPSILON, "Softmax 概率之和应为 1");
    }

    @Test
    void testSoftmaxMaxElementHasHighestProb() {
        Value[] logits = new Value[]{new Value(1.0), new Value(5.0), new Value(2.0)};
        Value[] probs = model.softmax(logits);
        assertTrue(probs[1].data > probs[0].data, "最大 logit 对应的概率应最高");
        assertTrue(probs[1].data > probs[2].data, "最大 logit 对应的概率应最高");
    }

    @Test
    void testSoftmaxNumericalStability() {
        // 测试大数值不会导致溢出
        Value[] logits = new Value[]{new Value(1000.0), new Value(1001.0), new Value(1002.0)};
        Value[] probs = model.softmax(logits);

        double sum = 0;
        for (Value p : probs) {
            assertFalse(Double.isNaN(p.data), "大数值 softmax 不应产生 NaN");
            sum += p.data;
        }
        assertEquals(1.0, sum, EPSILON);
    }

    @Test
    void testRmsnorm() {
        Value[] input = new Value[]{new Value(3.0), new Value(4.0)};
        Value[] output = model.rmsnorm(input);

        assertEquals(input.length, output.length, "RMS Norm 输出维度应与输入一致");

        // RMS Norm 后向量的均方根应接近 1
        double sumSquares = 0;
        for (Value v : output) {
            sumSquares += v.data * v.data;
            assertFalse(Double.isNaN(v.data), "RMS Norm 输出不应为 NaN");
        }
        double rms = Math.sqrt(sumSquares / output.length);
        assertEquals(1.0, rms, 0.01, "RMS Norm 后的均方根应接近 1");
    }

    @Test
    void testLinear() {
        Value[] input = new Value[]{new Value(1.0), new Value(2.0)};
        Value[][] weights = new Value[][]{
                {new Value(1.0), new Value(0.0)},
                {new Value(0.0), new Value(1.0)},
                {new Value(1.0), new Value(1.0)}
        };
        Value[] output = model.linear(input, weights);

        assertEquals(3, output.length, "线性变换输出维度应等于权重行数");
        assertEquals(1.0, output[0].data, EPSILON, "恒等变换第一维");
        assertEquals(2.0, output[1].data, EPSILON, "恒等变换第二维");
        assertEquals(3.0, output[2].data, EPSILON, "求和变换");
    }

    @Test
    void testInitKVCache() {
        List<List<Value[]>> cache = model.initKVCache();
        assertEquals(N_LAYER, cache.size(), "KV Cache 层数应等于模型层数");
        for (List<Value[]> layer : cache) {
            assertTrue(layer.isEmpty(), "初始 KV Cache 应为空");
        }
    }

    @Test
    void testStateDictNotEmpty() {
        assertNotNull(model.getStateDict());
        assertFalse(model.getStateDict().isEmpty(), "状态字典不应为空");
    }
}
