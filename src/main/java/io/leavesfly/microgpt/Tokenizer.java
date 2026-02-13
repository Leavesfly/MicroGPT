package io.leavesfly.microgpt;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Tokenizer 类 - 字符级分词器
 * 
 * 将文本转换为离散符号（token），并支持反向解码。
 */
public class Tokenizer {
    /** BOS (Beginning of Sequence) 特殊标记 */
    public static final String BOS_TOKEN = "<BOS>";
    
    /** 字符到索引的映射（编码） */
    private Map<Character, Integer> stoi;
    
    /** 索引到字符的映射（解码） */
    private Map<Integer, Character> itos;
    
    /** 词表大小 */
    private int vocabSize;
    
    /** BOS 标记的索引 */
    private int BOS;
    
    /** 所有字符列表 */
    private List<Character> chars;
    
    /**
     * 构造函数 - 基于文档列表构建分词器
     * @param docs 文档列表，用于构建词表
     */
    public Tokenizer(List<String> docs) {
        buildVocabulary(docs);
    }
    
    /**
     * 构建词表
     * 收集所有文档中的唯一字符，并添加 BOS 特殊标记
     */
    private void buildVocabulary(List<String> docs) {
        // 收集所有唯一字符
        Set<Character> charSet = new TreeSet<>();
        for (String doc : docs) {
            for (char c : doc.toCharArray()) {
                charSet.add(c);
            }
        }
        
        // 构建排序后的字符列表
        chars = new ArrayList<>(charSet);
        
        // stoi: string to integer (编码映射)
        stoi = new HashMap<>();
        // itos: integer to string (解码映射)
        itos = new HashMap<>();
        
        // BOS 标记放在索引0的位置
        stoi.put('\0', 0); // 使用空字符表示 BOS
        itos.put(0, '\0');
        
        // 其他字符从索引1开始
        int idx = 1;
        for (Character c : chars) {
            if (!stoi.containsKey(c)) {
                stoi.put(c, idx);
                itos.put(idx, c);
                idx++;
            }
        }
        
        vocabSize = stoi.size();
        BOS = 0; // BOS 标记的索引为 0
        
        System.out.println("词表大小: " + vocabSize);
    }
    
    /**
     * 编码单个字符
     * @param c 要编码的字符
     * @return 字符对应的索引
     */
    public int encode(char c) {
        return stoi.getOrDefault(c, 0);
    }
    
    /**
     * 编码字符串
     * @param s 要编码的字符串
     * @return 字符索引数组
     */
    public int[] encode(String s) {
        int[] tokens = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            tokens[i] = encode(s.charAt(i));
        }
        return tokens;
    }
    
    /**
     * 解码单个索引
     * @param idx 要解码的索引
     * @return 索引对应的字符
     */
    public char decode(int idx) {
        return itos.getOrDefault(idx, '\0');
    }
    
    /**
     * 解码索引数组
     * @param tokens 要解码的索引数组
     * @return 解码后的字符串
     */
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            char c = decode(token);
            if (c != '\0') {
                sb.append(c);
            }
        }
        return sb.toString();
    }
    
    /**
     * 获取 BOS 标记的索引
     */
    public int getBOS() {
        return BOS;
    }
    
    /**
     * 获取词表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }
    
    /**
     * 获取编码映射表
     */
    public Map<Character, Integer> getStoi() {
        return stoi;
    }
    
    /**
     * 获取解码映射表
     */
    public Map<Integer, Character> getItos() {
        return itos;
    }
    
    /**
     * 从 classpath resources 加载数据集
     *
     * @param resourceName resources 目录下的文件名（如 "input.txt"）
     * @return 文档列表
     */
    public static List<String> loadDataset(String resourceName) {
        List<String> docs = new ArrayList<>();

        try (InputStream inputStream = Tokenizer.class.getClassLoader().getResourceAsStream(resourceName)) {
            if (inputStream == null) {
                System.err.println("资源文件未找到: " + resourceName);
                return docs;
            }
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty()) {
                        docs.add(line);
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("读取资源文件失败: " + e.getMessage());
        }

        return docs;
    }
    
    /**
     * 打印词表信息
     */
    public void printVocabulary() {
        System.out.println("词表内容:");
        for (Map.Entry<Integer, Character> entry : itos.entrySet()) {
            if (entry.getKey() == 0) {
                System.out.println("  " + entry.getKey() + " -> <BOS>");
            } else {
                System.out.println("  " + entry.getKey() + " -> '" + entry.getValue() + "'");
            }
        }
    }
}
