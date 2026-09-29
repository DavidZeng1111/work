# DASE7506 Mini Project 1 最终模型复现说明

按照以下步骤可完整复现 **测试集 BPB = 1.765** 的结果，所有流程与官方基准一致，结果可稳定重复。

---

## 1. 环境要求

- Python 版本：3.12
- PyTorch 版本：2.7.1（CPU 版本，保证评测可复现）
- 操作系统：Linux /macOS/ Windows 均可，Linux 与官方参考环境最一致
- 无需额外下载数据、预训练权重或 API 密钥，所有资源已包含在项目包中

---

## 2. 环境安装步骤

所有命令均在项目的 `code/` 目录下执行。

### 步骤 1：创建并激活虚拟环境

```
cd code
python -m venv .venv

# Linux / macOS 激活
source .venv/bin/activate

# Windows PowerShell 激活
.venv\Scripts\Activate.ps1
```

### 步骤 2：安装 CPU 版 PyTorch

```
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

### 步骤 3：安装其余依赖

```
python -m pip install -r requirements.txt
```

### 步骤 4：合约合规性校验

运行官方单元测试，验证模型符合因果性、归一化、无状态泄漏、梯度可传等所有合约要求：

```
python -m unittest discover -s tests -v
```

所有测试项显示 `OK`，确认模型合规后再继续。

---

## 3. 模型训练

最终提交模型采用**架构改进 + 适度扩容**的方案：

- 架构改进：RMSNorm 归一化、SwiGLU 门控激活、QK 归一化、RoPE 旋转位置编码、深度残差缩放初始化、Dropout 正则化
- 模型配置：5 层 Transformer、160 隐藏维度、5 个注意力头
- 训练策略：warmup 余弦退火学习率、EMA 权重指数移动平均、AdamW 权重衰减、梯度裁剪
- 训练总 token 数：9,830,400（1200 步 × 32 batch size × 256 上下文），与基线完全一致

### 训练命令

```
python train.py \
  --implementation student \
  --config configs/wide.json \
  --device cpu \
  --threads 4 \
  --seed 17 \
  --steps 1200 \
  --eval-every 300 \
  --run-dir runs/final29
```

参数说明：

- `--implementation student`：使用 `student.py` 中的改进模型
- `--config configs/wide.json`：使用 5 层 160 维的扩容配置
- `--seed 17`：固定随机种子，保证可复现
- `--steps 1200`：与基线相同的训练步数，公平对比
- `--eval-every 300`：每 300 步输出一次验证集 BPB，便于观察收敛过程
- `--run-dir runs/final29`：输出目录，训练完成后生成模型权重与指标文件

训练完成后，`runs/final29/` 目录下会生成：

- `checkpoint.pt`：最终模型权重文件
- `metrics.json`：训练过程记录与验证集最终结果

---

## 4. 模型评估

### 最终测试集评估命令


```
python evaluate.py \
  --checkpoint runs/final29/checkpoint.pt \
  --device cpu \
  --precision fp32 \
  --split test
```

### 输出结果

运行后会生成 `test_cpu_fp32.json`，核心结果如下：

```
{
  "bpb": 1.7654946209156854,
  "token_ppl": 40.07134466457621,
  "nll_nats": 1581097.8321846863,
  "targets": 428405,
  "utf8_bytes": 1292013,
  "seconds": 6.933065916004125,
  "protocol": "7506-mp1-wt2-v2",
  "split": "test",
  "precision": "fp32"
}
```

核心评分：**测试集 BPB = 1.765**
