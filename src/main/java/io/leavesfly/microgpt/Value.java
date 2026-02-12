package io.leavesfly.microgpt;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Value 类 - 存储单个标量值及其梯度
 * 
 * 这是一个简化的自动微分引擎，支持基本的数学运算和反向传播。
 */
public class Value {
    /** 存储的实际数据值 */
    public double data;
    
    /** 存储梯度值，在反向传播时计算 */
    public double grad;
    
    /** 反向传播函数，用于计算梯度 */
    private Consumer<Void> backward;
    
    /** 计算图中的前驱节点（子节点） */
    private Set<Value> prev;
    
    /** 产生此节点的操作名称，用于调试和可视化 */
    private String op;
    
    /**
     * 构造函数
     * @param data 初始数据值
     */
    public Value(double data) {
        this(data, new ArrayList<>(), "");
    }
    
    /**
     * 构造函数
     * @param data 初始数据值
     * @param children 子节点列表
     * @param op 操作名称
     */
    public Value(double data, List<Value> children, String op) {
        this.data = data;
        this.grad = 0.0;
        this.backward = (v) -> {};
        this.prev = new HashSet<>(children);
        this.op = op;
    }
    
    /**
     * 加法运算
     */
    public Value add(Value other) {
        if (other == null) {
            other = new Value(0);
        }
        Value out = new Value(this.data + other.data, List.of(this, other), "+");
        
        // 定义反向传播函数
        final Value finalOther = other;
        Consumer<Void> backwardFn = (v) -> {
            this.grad += out.grad;
            finalOther.grad += out.grad;
        };
        out.setBackward(backwardFn);
        
        return out;
    }
    
    /**
     * 加法运算（支持 double 类型）
     */
    public Value add(double other) {
        return add(new Value(other));
    }
    
    /**
     * 乘法运算
     */
    public Value mul(Value other) {
        if (other == null) {
            other = new Value(0);
        }
        Value out = new Value(this.data * other.data, List.of(this, other), "*");
        
        // 定义反向传播函数
        final Value finalOther = other;
        Consumer<Void> backwardFn = (v) -> {
            this.grad += finalOther.data * out.grad;
            finalOther.grad += this.data * out.grad;
        };
        out.setBackward(backwardFn);
        
        return out;
    }
    
    /**
     * 乘法运算（支持 double 类型）
     */
    public Value mul(double other) {
        return mul(new Value(other));
    }
    
    /**
     * 幂运算
     * @param other 指数（仅支持 int 或 float）
     */
    public Value pow(double other) {
        Value out = new Value(Math.pow(this.data, other), List.of(this), "**" + other);
        
        // 定义反向传播函数
        Consumer<Void> backwardFn = (v) -> {
            this.grad += (other * Math.pow(this.data, other - 1)) * out.grad;
        };
        out.setBackward(backwardFn);
        
        return out;
    }
    
    /**
     * 自然对数运算
     */
    public Value log() {
        // 添加数值稳定性保护，防止 log(0)
        double safeData = Math.max(this.data, 1e-10);
        Value out = new Value(Math.log(safeData), List.of(this), "log");
        
        // 定义反向传播函数
        Consumer<Void> backwardFn = (v) -> {
            this.grad += (1.0 / safeData) * out.grad;
        };
        out.setBackward(backwardFn);
        
        return out;
    }
    
    /**
     * 指数运算
     */
    public Value exp() {
        // 添加数值稳定性保护，防止溢出
        double safeData = Math.min(this.data, 50.0);  // exp(50) ≈ 5e21
        Value out = new Value(Math.exp(safeData), List.of(this), "exp");
        
        // 定义反向传播函数
        Consumer<Void> backwardFn = (v) -> {
            this.grad += out.data * out.grad;
        };
        out.setBackward(backwardFn);
        
        return out;
    }
    
    /**
     * Sigmoid 激活函数
     * σ(x) = 1 / (1 + exp(-x))
     */
    public Value sigmoid() {
        double sigVal = 1.0 / (1.0 + Math.exp(-this.data));
        Value out = new Value(sigVal, List.of(this), "sigmoid");

        Consumer<Void> backwardFn = (v) -> {
            this.grad += sigVal * (1.0 - sigVal) * out.grad;
        };
        out.setBackward(backwardFn);

        return out;
    }

    /**
     * 截断计算图（detach）
     * 返回一个与当前值相同但不参与梯度计算的新 Value 节点
     * 用于 PPO 中冻结 reference model 的输出
     */
    public Value detach() {
        return new Value(this.data);
    }

    /**
     * ReLU 激活函数
     */
    public Value relu() {
        Value out = new Value(this.data < 0 ? 0 : this.data, List.of(this), "ReLU");
        
        // 定义反向传播函数
        Consumer<Void> backwardFn = (v) -> {
            this.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
        };
        out.setBackward(backwardFn);
        
        return out;
    }
    
    /**
     * 反向传播 - 计算所有参数的梯度
     */
    public void backward() {
        // 构建拓扑排序
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, topo, visited);
        
        // 设置初始梯度为1
        this.grad = 1.0;
        
        // 按拓扑逆序应用链式法则
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i).getBackward().accept(null);
        }
    }
    
    /**
     * 构建拓扑排序
     * 深度优先遍历计算图，确保子节点先于父节点被处理
     */
    private void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.prev) {
                buildTopo(child, topo, visited);
            }
            topo.add(v);
        }
    }
    
    // ============ 辅助运算方法 ============

    /**
     * 取负运算
     */
    public Value neg() {
        return this.mul(-1.0);
    }

    /**
     * 减法运算（原生实现，避免中间节点）
     */
    public Value sub(Value other) {
        Value out = new Value(this.data - other.data, List.of(this, other), "-");

        Consumer<Void> backwardFn = (v) -> {
            this.grad += out.grad;
            other.grad -= out.grad;
        };
        out.setBackward(backwardFn);

        return out;
    }

    /**
     * 减法运算（支持 double 类型）
     */
    public Value sub(double other) {
        return sub(new Value(other));
    }

    /**
     * 除法运算（原生实现，避免中间节点）
     */
    public Value div(Value other) {
        Value out = new Value(this.data / other.data, List.of(this, other), "/");

        Consumer<Void> backwardFn = (v) -> {
            this.grad += (1.0 / other.data) * out.grad;
            other.grad -= (this.data / (other.data * other.data)) * out.grad;
        };
        out.setBackward(backwardFn);

        return out;
    }

    /**
     * 除法运算（支持 double 类型，直接用标量除，只产生 1 个节点）
     */
    public Value div(double other) {
        Value out = new Value(this.data / other, List.of(this), "/" + other);

        Consumer<Void> backwardFn = (v) -> {
            this.grad += (1.0 / other) * out.grad;
        };
        out.setBackward(backwardFn);

        return out;
    }
    
    // ============ Getter 和 Setter ============
    
    public Consumer<Void> getBackward() {
        return backward;
    }
    
    public void setBackward(Consumer<Void> backward) {
        this.backward = backward;
    }
    
    public Set<Value> getPrev() {
        return prev;
    }
    
    public String getOp() {
        return op;
    }
    
    @Override
    public String toString() {
        return String.format("Value(data=%.6f, grad=%.6f)", data, grad);
    }
}
