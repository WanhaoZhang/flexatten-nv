# FlexAttention + FlashAttention 验证报告

## 1. 项目概述

本项目在 **NVIDIA L4 GPU** 上完成了 Meta PyTorch **FlexAttention** API 与 **FlashAttention** 后端的完整验证流程。基于 [attention-gym](https://github.com/meta-pytorch/attention-gym) 仓库，测试了 FlexAttention 的多种注意力掩码模式、分数修改（score modification）能力，并与 PyTorch SDPA（Scaled Dot Product Attention）的 FlashAttention2 后端进行了性能对比。

**测试环境：**
- 服务器：GCP L4 实例（`136.115.44.140`）
- GPU：NVIDIA L4（22GB VRAM）
- OS：Debian 11 (bullseye)
- CUDA Driver：12.4（550.90.07）

---

## 2. 环境搭建

### 2.1 软件依赖

| 组件 | 版本 |
|------|------|
| Python | 3.11（Miniconda） |
| PyTorch | 2.6.0+cu124 |
| CUDA Runtime | 12.4 |
| cuDNN | 9.1.0 |
| Triton | 3.2.0 |
| attention-gym | 0.0.5.dev38 |

### 2.2 安装步骤

```bash
# 1. 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 2. 创建虚拟环境
conda create -y -n flexatten python=3.11
conda activate flexatten

# 3. 安装 PyTorch（CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 克隆并安装 attention-gym
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym
pip install -e ".[dev,viz]"
```

### 2.3 环境验证

```
PyTorch 版本:       2.6.0+cu124
CUDA 版本:          12.4
cuDNN 版本:         90100
GPU 名称:           NVIDIA L4
GPU 显存:           22.0 GB
Flash SDPA:         True
Mem Efficient SDPA: True
Math SDPA:          True
cuDNN SDPA:         True
```

---

## 3. FlexAttention 调用链路分析

### 3.1 核心架构

FlexAttention 的设计目标是让用户通过 **Python 级别的 mask 函数** 和 **score 修改函数** 来定义任意注意力模式，而无需编写自定义 CUDA kernel。其核心调用链路如下：

```
用户定义 mask_mod / score_mod
        │
        ▼
create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN)
        │
        ├── 使用 vmap 对 mask_mod 进行向量化求值
        ├── 生成 BlockMask（稀疏块级掩码表示）
        │   ├── kv_num_blocks / kv_indices: KV 侧的块级索引
        │   ├── q_num_blocks / q_indices: Q 侧的块级索引
        │   └── BLOCK_SIZE: 默认 (128, 128)
        │
        ▼
flex_attention(q, k, v, score_mod=None, block_mask=None)
        │
        ├── torch.compile() 编译为 Triton kernel
        │   ├── Score 修改 → Triton score_mod kernel
        │   ├── Block mask → Triton 稀疏 attention kernel
        │   └── 自动融合为单个 kernel
        │
        ▼
GPU 执行（Triton kernel → CUDA）
```

### 3.2 关键组件详解

#### `create_block_mask()`
- **输入**：用户定义的 Python 函数 `mask_mod(b, h, q_idx, kv_idx) -> bool`
- **处理**：使用 `torch.vmap` 对 mask 函数进行 4 层向量化（batch, head, q, kv），生成完整的 boolean mask 矩阵
- **输出**：`BlockMask` 对象，将 boolean mask 压缩为块级稀疏表示
  - `BLOCK_SIZE = (128, 128)`：将 Q 和 KV 维度划分为 128x128 的块
  - 如果一个块内所有位置都是 True（可见），标记为 full block
  - 如果一个块内所有位置都是 False（不可见），直接跳过
  - 只保留部分可见的块信息，大幅减少计算量

#### `flex_attention()`
- **核心**：`flex_attention = torch.compile(_flex_attention_impl)`
- **编译过程**：
  1. PyTorch Dynamo 捕获计算图
  2. 将 `score_mod` 和 `mask_mod` 编译为 Triton kernel
  3. Triton 生成 CUDA PTX 代码
- **执行策略**：
  - 每个 attention head 独立处理
  - 利用 BlockMask 的稀疏性跳过不需要计算的块
  - 支持反向传播（autograd）

#### 与 FlashAttention 的关系
- **SDPA**（`F.scaled_dot_product_attention`）是 PyTorch 内置的高效注意力实现
- SDPA 内部有 4 个后端：FlashAttention2、Memory Efficient、Math、cuDNN
- **FlexAttention 不是 SDPA 的替代品**，而是提供了 SDPA 无法实现的**自定义注意力模式**
- FlexAttention 通过 Triton 编译实现自定义 kernel，灵活但性能略低于 FlashAttention2 的手写 CUDA kernel

---

## 4. 测试结果

### 4.1 测试项汇总

| # | 测试项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | Causal Mask | PASS | 标准因果注意力掩码 |
| 2 | Sliding Window | PASS | 窗口大小 32 的滑动注意力 |
| 3 | Document Mask | PASS | 8 个文档（每 64 tokens）的独立注意力 |
| 4 | 性能对比 | PASS | FlexAttention vs SDPA 对比 |
| 5 | Prefix LM Mask | PASS | 前缀 LM 掩码（前缀双向 + 后缀因果） |
| 6 | ALiBi Score Mod | PASS | ALiBi 位置偏置分数修改 |
| 7 | Flash 后端验证 | PASS | Flash/MemEff/Math 后端均可用 |
| 8 | 大规模压力测试 | PASS | 2048/4096/8192 序列长度 |

### 4.2 性能数据

#### FlexAttention vs SDPA（Causal, B=1, H=8, D=64）

| 序列长度 | SDPA (ms) | FlexAttention (ms) | 比率 | GPU 显存 |
|----------|-----------|-------------------|------|----------|
| 1024 | 0.068 | 0.229 | 3.39x | - |
| 2048 | 0.17 | 0.45 | 2.66x | 0.30 GB |
| 4096 | 0.37 | 0.64 | 1.74x | 1.17 GB |
| 8192 | 1.07 | 1.47 | 1.37x | 4.61 GB |

**关键发现：**
- FlexAttention 在短序列上有较大开销（编译 + Triton kernel 启动），约为 SDPA 的 3.4x
- 随着序列长度增加，性能差距缩小，8192 长度时仅慢 1.37x
- FlexAttention 的核心优势在于**灵活性**：可以用同一 API 实现任意注意力模式
- FlashAttention2 仍然是标准因果注意力的最优选择

#### 数值精度
- SDPA（FlashAttention2）与 FlexAttention 的最大误差：**0.000488**（FP16）
- 误差来源于不同 kernel 的浮点运算顺序差异，在 FP16 精度下属于正常范围

### 4.3 各注意力模式说明

#### Causal Mask（因果掩码）
```python
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx  # 每个位置只能看到自己和之前的位置
```
标准自回归注意力，用于 GPT 等自回归语言模型。

#### Sliding Window Attention（滑动窗口注意力）
```python
def sliding_window_mask(b, h, q_idx, kv_idx):
    return torch.abs(q_idx - kv_idx) <= WINDOW_SIZE
```
每个位置只关注固定窗口内的 token，用于 Mistral、Longformer 等模型，降低长序列的计算复杂度。

#### Document Mask（文档掩码）
```python
def document_mask(b, h, q_idx, kv_idx):
    return doc_ids[q_idx] == doc_ids[kv_idx]  # 同文档内可见
```
打包多个独立文档到同一序列时，防止跨文档注意力泄漏。用于批量推理场景。

#### Prefix LM Mask（前缀 LM 掩码）
```python
def prefix_lm_mask(b, h, q_idx, kv_idx):
    prefix_mask = kv_idx < PREFIX_LEN
    causal_mask = q_idx >= kv_idx
    return prefix_mask | causal_mask  # 前缀部分双向，其余因果
```
前缀部分（如系统提示）使用双向注意力，后续 token 使用因果注意力。用于 T5、UniLM 等 prefix-LM 模型。

#### ALiBi Score Modification（ALiBi 分数修改）
```python
def alibi_score_mod(score, b, h, q_idx, kv_idx):
    slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625])
    return score - slopes[h] * (q_idx - kv_idx).abs()
```
在 attention score 上加入距离惩罚（而非位置编码），用于 BLOOM、MPT 等模型，提升长度外推能力。

---

## 5. 调用链路详解

### 5.1 端到端执行流程

```
[用户代码]
    │
    ├── 1. 定义 mask_mod / score_mod（Python lambda/函数）
    │
    ├── 2. create_block_mask(mask_mod, B, H, S, S, device="cuda")
    │       │
    │       ├── vmap(mask_mod, [None, None, 0, None])  # 对 q_idx 向量化
    │       ├── vmap(vmap(mask_mod, ...), ...)          # 4 层 vmap 展开
    │       ├── 生成 (B, H, Q_LEN, KV_LEN) boolean tensor
    │       ├── 分块：按 BLOCK_SIZE=(128,128) 切分
    │       ├── 计算每个块的稀疏性
    │       └── 返回 BlockMask（压缩后的块级掩码）
    │
    ├── 3. flex_attention(q, k, v, score_mod, block_mask)
    │       │
    │       ├── [首次调用] torch.compile 触发编译
    │       │   ├── Dynamo 捕获计算图
    │       │   ├── Inductor 将 score_mod 编译为 Triton kernel
    │       │   ├── Inductor 将 mask 应用逻辑编译为 Triton kernel
    │       │   └── Triton 生成 CUDA PTX → 缓存
    │       │
    │       └── [执行]
    │           ├── Q·K^T → Triton score_mod kernel
    │           ├── Block mask 稀疏过滤
    │           ├── Softmax（逐行归一化）
    │           ├── × V → 输出
    │           └── 支持 autograd 反向传播
    │
    └── 4. 返回 output tensor (B, H, S, D)
```

### 5.2 FlashAttention 在 SDPA 中的调用链路

```
F.scaled_dot_product_attention(q, k, v, is_causal=True)
    │
    ├── SDPA 后端选择逻辑
    │   ├── 优先级: Flash > MemEfficient > cuDNN > Math
    │   └── 根据 dtype, shape, CUDA capability 自动选择
    │
    ├── [FlashAttention2 后端]
    │   ├── 调用 flash_attn_cuda.varlen_fwd / fwd
    │   ├── 分块计算（tiling）: 将 Q, K, V 分成小块
    │   ├── Online softmax: 分块累积 softmax 归一化
    │   ├── 不显式构建 N×N attention 矩阵
    │   └── 内存复杂度 O(N) 而非 O(N²)
    │
    └── 返回 output tensor
```

### 5.3 FlexAttention 与 FlashAttention 的区别

| 特性 | FlexAttention | FlashAttention (via SDPA) |
|------|--------------|--------------------------|
| 自定义 mask | 任意 Python 函数 | 仅 causal / 特定模式 |
| 自定义 score_mod | 支持（ALiBi, soft-cap 等） | 不支持 |
| 编译方式 | Triton (JIT) | 手写 CUDA kernel |
| 性能（标准 causal） | 基准 | ~2-3x 更快 |
| 灵活性 | 极高 | 受限于预定义模式 |
| 编译开销 | 首次调用有编译 | 无 |
| 适用场景 | 非标准注意力模式 | 标准 attention 模式 |

---

## 6. 注意事项与已知限制

1. **PyTorch 版本兼容性**：attention-gym 的 `attn_gym.mods` 模块需要 PyTorch >= 2.7（`AuxRequest` API）。PyTorch 2.6.0 下可直接使用 `torch.nn.attention.flex_attention` 的基础 API。

2. **编译开销**：FlexAttention 首次调用会触发 Triton 编译（约 5-15 秒），后续调用使用缓存。生产环境需预热。

3. **FP16 精度**：FlexAttention 与 SDPA 在 FP16 下有约 0.0005 的数值差异，属于正常范围。

4. **序列长度限制**：L4 24GB 在 S=8192, B=1, H=8, D=64 下占用 4.6GB。序列长度受 GPU 显存约束。

5. **Triton 依赖**：FlexAttention 依赖 Triton 进行 JIT 编译，不支持 CPU only 环境。

---

## 7. 文件结构

```
flexatten-nv/
├── README.md              # 本报告
├── run_all_tests.py       # 完整测试脚本（8 项测试）
└── .gitignore
```

---

## 8. 快速复现

```bash
# 连接服务器
ssh -i ~/.ssh/gcp_l4 zhangwh@136.115.44.140

# 激活环境
conda activate flexatten

# 运行完整测试
python ~/flexatten-nv-tests.py
```

---

*报告生成时间：2026-04-22*
*GPU：NVIDIA L4 (22GB) | PyTorch 2.6.0+cu124 | CUDA 12.4*
