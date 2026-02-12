package io.leavesfly.microgpt;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tokenizer 分词器测试
 */
class TokenizerTest {

    private Tokenizer tokenizer;
    private List<String> docs;

    @BeforeEach
    void setUp() {
        docs = Arrays.asList("hello", "world", "java");
        tokenizer = new Tokenizer(docs);
    }

    @Test
    void testVocabSizeIncludesBos() {
        // 词表应包含 BOS + 所有唯一字符
        // "hello", "world", "java" 的唯一字符: a, d, e, h, j, l, o, r, v, w = 10 个
        // 加上 BOS = 11
        assertEquals(11, tokenizer.getVocabSize());
    }

    @Test
    void testBosIndex() {
        assertEquals(0, tokenizer.getBOS(), "BOS 标记的索引应为 0");
    }

    @Test
    void testEncodeDecodeConsistency() {
        String text = "hello";
        for (char c : text.toCharArray()) {
            int encoded = tokenizer.encode(c);
            char decoded = tokenizer.decode(encoded);
            assertEquals(c, decoded, "编码后解码应得到原始字符: " + c);
        }
    }

    @Test
    void testEncodeString() {
        int[] tokens = tokenizer.encode("hel");
        assertEquals(3, tokens.length);
        assertEquals('h', tokenizer.decode(tokens[0]));
        assertEquals('e', tokenizer.decode(tokens[1]));
        assertEquals('l', tokenizer.decode(tokens[2]));
    }

    @Test
    void testDecodeTokenArray() {
        int[] tokens = tokenizer.encode("java");
        String decoded = tokenizer.decode(tokens);
        assertEquals("java", decoded);
    }

    @Test
    void testUnknownCharEncodesToBos() {
        int encoded = tokenizer.encode('z');
        assertEquals(tokenizer.getBOS(), encoded, "未知字符应编码为 BOS 索引");
    }

    @Test
    void testAllCharsHaveUniqueIndices() {
        // "hello", "world", "java" 的唯一字符: a, d, e, h, j, l, o, r, v, w
        String allChars = "adehjlorvw";
        int[] indices = new int[allChars.length()];
        for (int i = 0; i < allChars.length(); i++) {
            indices[i] = tokenizer.encode(allChars.charAt(i));
        }
        // 检查所有索引都不同
        for (int i = 0; i < indices.length; i++) {
            for (int j = i + 1; j < indices.length; j++) {
                assertNotEquals(indices[i], indices[j],
                        "字符 '" + allChars.charAt(i) + "' 和 '" + allChars.charAt(j) + "' 的索引不应相同");
            }
        }
    }

    @Test
    void testDecodeBosReturnsNullChar() {
        char decoded = tokenizer.decode(tokenizer.getBOS());
        assertEquals('\0', decoded, "解码 BOS 索引应返回空字符");
    }
}
