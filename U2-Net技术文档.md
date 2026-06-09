# U²-Net 技术实现原理与架构详解

## 一、概述

U²-Net（U Square Net）是发表在 Pattern Recognition 2020 上的显著性目标检测（Salient Object Detection, SOD）模型，核心创新在于提出了 **嵌套 U 结构（Nested U-Structure）**，在单个网络中同时捕获多尺度特征，兼具大感受野和高分辨率的优势。

---

## 二、核心架构：嵌套 U 结构

### 2.1 RSU 模块（ReSidual U-block）

RSU 是 U²-Net 的基本构建块，结构类似一个微型 U-Net，包含：

- **输入卷积**：`REBNCONV(in_ch, out_ch)` — 将输入映射到输出通道
- **编码器**：多个下采样阶段（MaxPool2d + REBNCONV），逐步降低空间分辨率
- **桥接层**：最底层的膨胀卷积（dilation=2），扩大感受野
- **解码器**：通过上采样 + 跳跃连接还原空间尺寸
- **残差连接**：最终输出 = 解码结果 + 输入卷积结果（`hx1d + hxin`）

RSU 有 5 种变体，区别在于下采样层数：

| RSU 变体 | 下采样次数 | 等效深度 | 用途 |
|---------|----------|---------|------|
| RSU-7   | 5 次 (×32) | 7 层 | U²-Net Stage 1 |
| RSU-6   | 4 次 (×16) | 6 层 | U²-Net Stage 2 |
| RSU-5   | 3 次 (×8)  | 5 层 | U²-Net Stage 3 |
| RSU-4   | 2 次 (×4)  | 4 层 | U²-Net Stage 4 |
| RSU-4F  | 0 次（膨胀卷积） | 4 层 | U²-Net Stage 5/6，不降分辨率 |

### 2.2 基础组件：REBNCONV

REBNCONV = **Re**LU + **B**atch**N**orm + **CONV**，是 RSU 内的基本运算单元：

```
Conv2d(ReLU(BatchNorm(x)))
```

所有卷积使用 3×3 核，通过 dilation 参数控制感受野。

### 2.3 整体网络结构

U²-Net 采用 6 阶段编码器 + 5 阶段解码器 + 多侧输出：

```
输入 (3 通道 RGB) → [Stage 1] → [Stage 2] → [Stage 3] → [Stage 4] → [Stage 5] → [Stage 6]
                        ↓           ↓           ↓           ↓           ↓           ↓
                      Pool(×2)    Pool(×2)    Pool(×2)    Pool(×2)    Pool(×2)    Upsample
                        ↓           ↓           ↓           ↓           ↓           ↓
                      [Stage 1d] ← [Stage 2d] ← [Stage 3d] ← [Stage 4d] ← [Stage 5d] ←
                         ↓           ↓           ↓           ↓           ↓           ↓
                       Side1       Side2       Side3       Side4       Side5       Side6
                         ↓           ↓           ↓           ↓           ↓           ↓
                       Up×1        Up×2        Up×4        Up×8        Up×16       Up×16
                         ↓           ↓           ↓           ↓           ↓           ↓
                       └─────────── 拼接 6 路侧输出 ───────────────────────────────┘
                                             ↓
                                       1×1 Conv (6→1)
                                             ↓
                                        Sigmoid 输出
```

### 2.4 多侧输出与深度监督

U²-Net 同时输出 7 个显著性图：

- **d1 ~ d6**：来自 6 个解码阶段的侧输出（Side Output），都被上采样到同一尺寸
- **d0**：所有侧输出拼接后经过 1×1 卷积融合的最终输出

训练时，使用 7 个 BCE 损失之和作为总损失（深度监督）：

```python
loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
```

推理时通常只使用 d0（融合输出），或者 d1（第一层侧输出）。

---

## 三、U²-Net vs U²-NetP（轻量版）

| 对比维度 | U²-Net（标准版） | U²-NetP（轻量版） |
|---------|----------------|-----------------|
| 权重文件大小 | ~176.3 MB | ~4.7 MB（缩小 37 倍） |
| 所有阶段 mid_ch | 32, 32, 64, 128, 256, 256 | **全部 16** |
| 编码器输出通道 | 64, 128, 256, 512, 512, 512 | **全部 64** |
| 解码器输入通道 | 逐级递增 | **全部 128** |
| 推理速度 | 较慢 | 快数倍 |
| 精度 | 高 | 略低但可接受 |

U²-NetP 的核心思想是在不改变架构拓扑的前提下，**大幅削减所有层的通道数**，实现模型尺寸的极致压缩，适合边缘设备部署。

---

## 四、推理流程

### 4.1 数据预处理

1. 图像调整为 320×320 分辨率（保持宽高比，多余部分填充）
2. 像素值归一化到 [0,1]
3. RGB 通道标准化（ImageNet 均值标准差）：
   - R: (val - 0.485) / 0.229
   - G: (val - 0.456) / 0.224
   - B: (val - 0.406) / 0.225
4. 转换为 CHW 格式的 Tensor

### 4.2 模型推理

```python
d0, d1, d2, d3, d4, d5, d6 = net(input_tensor)
# d0 是融合输出（推荐使用）
pred = d0[:, 0, :, :]
pred = (pred - min) / (max - min)  # 归一化到 [0,1]
```

### 4.3 后处理

- 对输出进行 min-max 归一化到 [0, 1]
- 调整回原始图像尺寸（双线性插值）
- 乘以 255 保存为 PNG

---

## 五、CPU 兼容性

所有推理脚本均内置 CPU 兼容逻辑：

```python
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
```

- **模型仅在 CPU 上执行推理**：无需 GPU，可运行在任何 x86 机器上
- 训练脚本默认优先使用 GPU，但无 GPU 时也回退 CPU

---

## 六、运行指南

### 6.1 环境要求

```bash
pip install torch torchvision scikit-image pillow numpy gdown
```

### 6.2 下载预训练模型

```bash
# u2netp（4.7 MB，推荐快速验证）
python3 -c "import gdown; gdown.download('https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy', './saved_models/u2netp/u2netp.pth')"

# u2net（176.3 MB，全量版）
python3 -c "import gdown; gdown.download('https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ', './saved_models/u2net/u2net.pth')"
```

### 6.3 运行推理

**方式一：修改原脚本的 model_name**

编辑 `u2net_test.py`，将第 57 行改为：
```python
model_name='u2netp'  # 默认是 'u2net'
```

然后运行：
```bash
python u2net_test.py
```

**方式二：直接用命令行指定（推荐）**

```bash
# 使用 u2netp 模型推理 test_data/test_images/ 下的所有图片
sed "s/model_name='u2net'/model_name='u2netp'/" u2net_test.py | python

# 输出位置：test_data/u2netp_results/
```

### 6.4 用自己的图片推理

将图片放入 `test_data/test_images/` 目录，然后重复上述命令即可。支持的格式：jpg、jpeg、png。

### 6.5 训练自己的模型

```bash
python u2net_train.py
```

在 `u2net_train.py` 中修改 `model_name` 可选择训练 U²-Net 或 U²-NetP。

---

## 七、模型下载地址

| 模型 | 大小 | Google Drive | 百度网盘 |
|-----|------|-------------|---------|
| u2net.pth | 176.3 MB | [下载](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view) | 提取码 pf9k |
| u2netp.pth | 4.7 MB | [下载](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view) | 提取码 8xsi |

---

## 八、代码结构

```
U-2-Net/
├── model/
│   ├── __init__.py       # 导出 U2NET, U2NETP
│   └── u2net.py          # RSU 模块 + U2NET + U2NETP 定义
├── u2net_test.py          # 显著目标检测推理脚本
├── u2net_train.py         # 训练脚本（深度监督 + 多 BCE 损失）
├── data_loader.py         # 数据加载、预处理、数据增强
├── setup_model_weights.py # 模型权重自动下载脚本
├── saved_models/
│   ├── u2net/             # u2net.pth 存放位置
│   └── u2netp/            # u2netp.pth 存放位置
├── test_data/
│   └── test_images/       # 输入图片
│   └── u2netp_results/    # u2netp 输出结果
│   └── u2net_results/     # u2net 输出结果
└── README.md
```

---

## 九、关键设计要点

1. **嵌套 U 结构**：RSU 块内部是一个微型 U-Net，U²-Net 整体又是 U 形架构，故得名 U²-Net（U Square Net）
2. **残差 U 块**：RSU 借鉴残差连接思想，学习输入与输出的残差，加速收敛
3. **多尺度特征融合**：6 个侧输出覆盖从细节（浅层高分辨率）到语义（深层大感受野）的多级特征
4. **深度监督**：每个侧输出都参与损失计算，梯度直接回传到编码器各层，缓解梯度消失
5. **膨胀卷积（dilated convolution）**：RSU-4F 用膨胀卷积替代下采样，在不降低分辨率的同时扩大感受野
6. **任意尺寸输入**：通过 `F.upsample`（bilinear 模式）将解码器特征自适应调整尺寸，支持任意输入分辨率（预训练模型推荐 320×320）