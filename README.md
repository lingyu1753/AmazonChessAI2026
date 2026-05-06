# KataGomo Amazons - 亚马逊棋 AI 训练项目

基于 KataGo 框架的亚马逊棋（Game of the Amazons）强化学习自对弈训练系统。

---

## 一、环境要求

### 1. 硬件
- NVIDIA GPU（推荐 RTX 3050 6GB 及以上）
- 内存 16GB 及以上
- 磁盘空间 50GB 以上（训练数据会持续增长）

### 2. 软件

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.11+ | 训练脚本运行环境 |
| CUDA Toolkit | 12.4 | GPU 加速 |
| cuDNN | 9.x | 深度神经网络加速库 |
| CMake | 3.20+ | C++ 编译工具 |
| Visual Studio Build Tools | 2022 | Windows C++ 编译器 |
| Git | 任意版本 | 版本管理 |

### 3. Python 依赖包

```bash
pip install torch numpy pillow
```

> PyTorch 需安装 CUDA 版本：`pip install torch --index-url https://download.pytorch.org/whl/cu124`

---

## 二、项目结构

```
KataGomo/
├── cpp/                          # C++ 源码（KataGo 引擎）
│   └── build/Release/
│       └── katago.exe            # 编译产物：自对弈引擎
├── python/                       # Python 训练库
│   ├── train.py                  # 模型训练入口
│   ├── shuffle.py                # 训练数据混洗
│   ├── export_model_pytorch.py   # 模型导出
│   └── katago/                   # KataGo Python 模块
├── training/                     # 训练配置与脚本
│   ├── models/
│   │   └── bootstrap-10x10/      # 初始随机模型
│   ├── scripts/
│   │   ├── train.py              # ★ 主训练脚本
│   │   ├── cleanup.py            # 清理脚本
│   │   ├── bootstrap_model.py    # 生成初始模型
│   │   └── check_env.py          # 环境检查
│   ├── selfplay.cfg         # ★ 自对弈参数配置
│   └── output/                   # ★ 训练输出目录
│       └── run-001/              # 第 N 次运行的输出
│           ├── model.bin         # 可部署的模型文件
│           ├── model.ckpt        # PyTorch 检查点
│           └── log.txt           # 导出日志
└── venv/                         # Python 虚拟环境
```

---

## 三、参数配置

### 主训练参数

**文件**：`training/scripts/train.py`（脚本顶部 `Configuration` 区域）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SELFPLAY_HOURS` | `20` | 每轮自对弈时长（小时） |
| `TRAIN_EPOCHS` | `150` | 每轮训练轮数 |
| `BATCH_SIZE` | `128` | 训练批次大小 |

### 自对弈参数

**文件**：`training/selfplay.cfg`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bSizes` | `10` | 棋盘尺寸（10x10） |
| `numGameThreads` | `4` | 并行对局线程数 |
| `maxVisits` | `200` | MCTS 每步搜索次数 |
| `maxTime` | `30.0` | 每步最大思考时间（秒） |
| `maxRowsPerTrainFile` | `2000` | 每个训练文件行数 |
| `nnMaxBatchSize` | `8` | 神经网络推理批次 |
| `nnCacheSizePowerOfTwo` | `18` | NNCache 大小（2^18） |
| `maxMovesPerGame` | `200` | 单局最大步数 |
| `validationProp` | `0.05` | 验证集比例（5%） |

### Shuffle 参数

**文件**：`training/scripts/train.py`（`step_shuffle` 函数中）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `keep-target-rows` | `500000` | 滑动窗口目标行数 |
| `expand-window-per-row` | `1.5` | 窗口扩张系数 |
| `taper-window-exponent` | `0.3` | 新旧数据衰减指数（越小越均匀） |

### 邮件通知参数

**文件**：`training/scripts/train.py`（`EMAIL_CONFIG` 字典）

| 参数 | 说明 |
|------|------|
| `host` | SMTP 服务器地址 |
| `port` | SMTP 端口 |
| `username` | 发送方邮箱 |
| `password` | 邮箱授权码 |
| `to` | 接收通知的邮箱 |

---

## 四、启动训练

### 首次使用

```powershell
# 1. 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 2. 清理旧测试文件（仅首次需要）
python training/scripts/cleanup.py

# 3. 检查环境
python training/scripts/check_env.py

# 4. 开始训练
# 正常完整训练
python training/scripts/train.py
# 续跑：跳过自对弈（数据已生成）
python training/scripts/train.py --skip-selfplay
# 续跑：跳过自对弈+shuffle（直接训练）
python training/scripts/train.py --skip-selfplay --skip-shuffle --shuffle-dir g:\KataGomo\training\shuffleddata\iter-3
```

### 后续训练

```powershell
.\venv\Scripts\Activate.ps1
python training/scripts/train.py
```

每次运行会自动：
1. 读取上次训练的模型作为起点
2. 自对弈生成新数据
3. 混洗数据
4. 训练模型
5. 导出可部署的模型文件

### 注意事项

- **以管理员身份运行**：脚本会自动禁用系统睡眠，需要管理员权限
- **不要关闭终端**：训练可能持续 30+ 小时
- **确保磁盘空间充足**：每轮自对弈数据约 2-5GB

---

## 五、训练结果

### 输出目录

```
training/output/run-001/    # 第 1 次运行
training/output/run-002/    # 第 2 次运行
training/output/run-003/    # 第 3 次运行
...
```

每次运行生成：

| 文件 | 说明 |
|------|------|
| `model.bin` | 可部署的 KataGo 格式模型 |
| `model.ckpt` | PyTorch 检查点（用于继续训练） |
| `log.txt` | 导出日志 |
| `metadata.json` | 模型元数据 |

### 运行次数记录

`training/run_counter.txt` 记录当前运行次数，每次训练自动 +1。

---

## 六、邮件通知

训练过程中会自动发送 3 封邮件到 `2975194966@qq.com`：

1. **训练开始** — 包含运行编号、参数配置
2. **自对弈完成** — 自对弈结束，即将开始训练
3. **训练完成** — 训练成功导出，或失败报错

---

## 七、清理

```powershell
# 清理旧测试文件和无用脚本
python training/scripts/cleanup.py
```

清理内容：
- 旧测试模型（quick-sample）
- 旧 PowerShell 脚本
- 旧日志文件
- 旧自对弈数据
- 临时文件
- 重复的源码目录

## 八 时间

┌────────────┬──────────┬─────────────────────────────────┐
│   步骤     │   耗时   │            说明                  │
├────────────┼──────────┼─────────────────────────────────┤
│ 1. 准备模型 │  ~1 分钟 │ 生成/复制 bootstrap 模型          │
│ 2. 自对弈   │ 13.5小时 │ 6线程, maxVisits=800, ~3万行数据  │
│ 3. Shuffle │ ~10 分钟 │ 6进程处理，保留 ~9万训练行         │
│ 4. 训练     │ ~1.5 小时│ 80 epochs × 781 batch/epoch     │
│            │          │ 每 epoch ~1.1 分钟               │
│ 5. 导出     │  ~1 分钟 │ 复制 .ckpt 到 output 目录         │
├────────────┼──────────┼─────────────────────────────────┤
│   总计     │ ~15 小时 │ 每轮迭代约 15 小时               │
└────────────┴──────────┴─────────────────────────────────┘