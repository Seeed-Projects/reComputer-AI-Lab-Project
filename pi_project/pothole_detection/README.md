# Raspberry Pi 5 · Hailo-8 Pothole Detection

基于 YOLOv8n 的路面坑洞检测项目，运行在 Raspberry Pi 5 + Hailo-8 加速器上。
支持视频批量推理和 Web 实时预览，带 ROI 道路区域过滤、短时跟踪补框和累计事件计数。

---

## 兼容性

| 组件 | 版本 |
|------|------|
| 主板 | Raspberry Pi 5 (aarch64) |
| 加速器 | Hailo-8 M.2 Key M 2280 (`/dev/hailo0`) |
| HailoRT | 4.23.0 |
| Python | 3.11 |
| 操作系统 | Raspberry Pi OS 64-bit Bookworm |

> HEF 面向 **Hailo-8**，不兼容 Hailo-8L。

---

## 项目结构

```
RaspberryPi5-Hailo8-pothole-detection/
├── app/
│   └── infer_video_hailo.py     # 视频推理入口
├── configs/
│   ├── runtime.json             # 运行时配置（模型路径、阈值、ROI、跟踪）
│   └── device_target.json       # 目标设备规格
├── hailort-packages/
│   └── hailort-4.23.0-cp311-cp311-linux_aarch64.whl  # 本地 HailoRT wheel
├── input/
│   └── demo.mp4                 # 演示视频 (1280×720, 25fps, 375帧)
├── models/
│   ├── hef/
│   │   └── pothole_yolov8n_hailo8.hef   # 最终部署模型
│   ├── onnx/
│   │   └── pothole_yolov8n.onnx
│   └── source_pt/
│       └── pothole_yolov8n.pt
├── pothole_monitor/
│   ├── processor.py             # 帧处理流水线（检测→ROI→跟踪→绘制）
│   └── tracker.py               # 时序跟踪器 + 补框逻辑
├── runtime/
│   ├── hailo_detector.py        # HailoRT 同步推理封装
│   └── yolo_postprocess.py      # letterbox + Hailo NMS 解码
├── scripts/
│   ├── install_rpi5.sh          # 树莓派环境安装
│   ├── probe.sh                 # Hailo 设备探测
│   ├── run_demo.sh              # 视频推理快捷脚本
│   └── docker_run.sh            # Docker 容器运行
├── tools/
│   ├── probe_hailo.py           # Python 级 Hailo 探测
│   └── check_deployment.py      # 部署包静态校验（无硬件可运行）
├── tests/
│   └── test_runtime.py          # 无硬件单元测试
├── conversion/                  # 模型转换工具和日志（部署时不需要）
├── Dockerfile
├── web_detection.py             # FastAPI/MJPEG Web 预览
├── requirements.txt             # Docker 用完整依赖
├── requirements-rpi5.txt        # 树莓派裸机依赖（OpenCV/NumPy 来自系统）
└── README.md
```

---

## 模型信息

| 项目 | 值 |
|------|-----|
| 架构 | YOLOv8n |
| 类别数 | 1 (`pothole`) |
| 输入尺寸 | 640×640 RGB |
| 输入格式 | uint8 (量化) |
| 输出 | Hailo NMS (YXYX 坐标 + 置信度) |
| PT → ONNX → HEF | 见 `conversion/logs/pothole_yolov8n.log` |

### HEF 校验

```
文件: models/hef/pothole_yolov8n_hailo8.hef
SHA-256: 6acec07e677cf4ae54919e0dd7105799ea8a50f1f48bd72528a6550b611b6856
```

验证命令：
```bash
sha256sum models/hef/pothole_yolov8n_hailo8.hef
# 应输出: 6acec07e...6856
```

---

## 树莓派安装步骤

> **前提**：Raspberry Pi OS 64-bit Bookworm + Hailo-8 硬件已安装在 M.2 槽位。

```bash
# 1. 将项目文件夹拷贝到树莓派（例如 ~/pothole-detection）
scp -r RaspberryPi5-Hailo8-pothole-detection pi@<PI_IP>:~/

# 2. SSH 登录后进入项目目录
cd ~/RaspberryPi5-Hailo8-pothole-detection
chmod +x scripts/*.sh

# 3. 运行安装脚本（检查架构/Python版本，安装依赖和HailoRT wheel）
./scripts/install_rpi5.sh

# 4. 如果 Hailo PCIe 驱动是新装的，需要重启
sudo reboot
```

安装脚本会：
- 检查 `aarch64` 和 `Python 3.11`（不满足则报错退出）
- 安装系统依赖（ffmpeg、libgl1 等）
- 创建 `.venv` 虚拟环境
- 从项目自带的本地 wheel 安装 HailoRT（**不联网下载，不升级**）
- 检查 `/dev/hailo0` 是否存在

---

## Hailo 设备检查

安装完成后，运行探测脚本确认硬件就绪：

```bash
./scripts/probe.sh
```

该脚本依次执行：

| 命令 | 作用 |
|------|------|
| `hailortcli scan` | 扫描 PCIe 上的 Hailo 设备 |
| `hailortcli fw-control identify` | 读取固件版本和设备信息 |
| `hailortcli parse-hef models/hef/pothole_yolov8n_hailo8.hef` | 解析 HEF，打印网络输入/输出 tensor 结构 |
| `python tools/probe_hailo.py` | Python 级验证 pyHailoRT 和 `/dev/hailo0` |

预期输出包含 `Hailo devices: {'xxx'}` 和 HEF 的 input/output layer 信息。

---

## 视频推理命令

```bash
# 使用默认配置 (configs/runtime.json)
./scripts/run_demo.sh

# 或直接调用，覆盖参数
.venv/bin/python app/infer_video_hailo.py \
  --config configs/runtime.json \
  --source input/demo.mp4 \
  --output outputs/pothole_demo_hailo8.mp4 \
  --confidence 0.25 \
  --iou 0.55

# 仅验证配置（不加载 HailoRT）
.venv/bin/python app/infer_video_hailo.py --check-config
```

输出：
- 标注视频 → `outputs/pothole_demo_hailo8.mp4`
- 推理报告 → `outputs/pothole_demo_hailo8.json`（帧数、FPS、检测数、累计事件）

视频画面叠加信息：FPS、当前坑洞数、累计事件数。
- **橙色框** + 置信度 = Hailo 当前帧直接检测
- **黄色框** "pothole tracked" = 跟踪补框（当前帧未检测到但前几帧有）

---

## Web 服务命令

```bash
# 启动 Web 预览（默认端口 8000）
.venv/bin/python web_detection.py \
  --config configs/runtime.json \
  --source input/demo.mp4 \
  --host 0.0.0.0 --port 8000

# 仅验证配置
.venv/bin/python web_detection.py --check-config

# 不循环视频（播完即停）
.venv/bin/python web_detection.py --no-loop
```

浏览器打开 `http://<PI_IP>:8000`，页面显示：
- 实时 MJPEG 视频流
- 推理 FPS
- 当前可见坑洞数
- 累计事件数（去重后的坑洞总数）
- 帧计数和运行状态
- 错误信息（如有）

API 端点：
- `GET /` — Web 页面
- `GET /api/video_feed` — MJPEG 流
- `GET /api/status` — JSON 状态
- `GET /healthz` — 健康检查

---

## Docker 构建和运行命令

> 在树莓派上（或任何 arm64 主机）构建。Raspberry Pi 宿主驱动和固件须兼容 HailoRT 4.23.x。

### 拉取已发布镜像

```bash
sudo docker pull \
  ghcr.io/seeed-projects/recomputer-ai-lab-project/pothole_detection:latest
```

### 本地构建

```bash
# 在仓库根目录执行（使用 docker/hailo8/ 下的 Dockerfile）
sudo docker build \
  -f docker/hailo8/pothole_detection.dockerfile \
  -t pothole-detection:latest \
  pi_project/pothole_detection
```

### 运行容器

```bash
# 方式一：Web 预览
./scripts/docker_run.sh web

# 方式二：视频推理
./scripts/docker_run.sh video

# 方式三：交互式 shell
./scripts/docker_run.sh shell
```

`docker_run.sh` 会映射：
- `/dev/hailo0` → 容器内设备访问
- `input/` → 只读挂载
- `output/` → 读写挂载（结果持久化到宿主机）
- `configs/` → 只读挂载
- 宿主机 `libhailort.so` → 只读挂载（版本须匹配 4.23.x）

> **不在容器中安装 x86_64 DFC**。HEF 已预编译，直接随项目使用。

---

## 配置参数说明

`configs/runtime.json`：

```json
{
  "source": "input/demo.mp4",          // 输入视频路径/RTSP URL/摄像头索引
  "output": "outputs/pothole_demo_hailo8.mp4",  // 输出视频路径
  "loop_video": true,                 // Web 模式下视频循环播放
  "model": {
    "path": "models/hef/pothole_yolov8n_hailo8.hef",  // HEF 路径
    "imgsz": 640,                      // 输入尺寸（必须 640）
    "confidence": 0.18,               // 置信度阈值
    "iou": 0.55,                      // NMS IoU 阈值
    "input_mode": "uint8"             // 输入量化模式 (uint8/float32)
  },
  "road_roi": {
    "enabled": true,                   // 是否启用道路区域过滤
    "horizon_y": 0.18,                // 地平线 Y（画面上方比例）
    "horizon_left": 0.22,             // 地平线左侧 X（梯形上边）
    "horizon_right": 0.78             // 地平线右侧 X（梯形上边）
  },
  "temporal": {
    "match_iou": 0.18,                // 跟踪匹配 IoU 阈值
    "min_hits": 2,                    // 最少命中次数才输出（去抖）
    "max_missed": 3,                   // 最大丢失帧数（超过则删除）
    "strong_confidence": 0.55         // 高置信度直接输出（不受 min_hits 限制）
  },
  "codec": "mp4v"                      // 输出视频编码
}
```

命令行可覆盖 `--source`、`--output`、`--confidence`、`--iou`。

---

## 黄色跟踪框说明

| 框颜色 | 含义 | 触发条件 |
|--------|------|----------|
| **橙色** `(255,90,20)` | Hailo 当前帧直接检测到 | 本帧 NMS 输出有检测框，经 ROI 过滤后匹配到已有 track |
| **黄色** `(0,200,255)` | 跟踪补框（gap-fill） | 该 track 在前几帧被检测到，但本帧未检测到（missed ≤ max_missed） |

补框逻辑：
1. 跟踪器为每个检测框分配唯一 ID
2. 连续帧之间用 IoU 匹配，匹配成功则更新位置
3. 如果某帧未检测到但 track 的 `missed` 未超过 `max_missed`（默认 3 帧），继续显示黄色框
4. 超过 `max_missed` 则删除该 track
5. 新 track 需达到 `min_hits`（默认 2 次）或置信度超过 `strong_confidence` 才显示

这样可以避免单帧漏检导致的闪烁，同时不会无限保留已离开画面的坑洞。

---

## 常见错误处理

### 1. `pyHailoRT is unavailable`
```
RuntimeError: pyHailoRT is unavailable; install the matching HailoRT 4.23 package
```
**原因**：虚拟环境未安装 HailoRT wheel。
**解决**：重新运行 `./scripts/install_rpi5.sh`，或手动：
```bash
.venv/bin/pip install hailort-packages/hailort-4.23.0-cp311-cp311-linux_aarch64.whl
```

### 2. `/dev/hailo0 not found`
**原因**：Hailo PCIe 驱动未安装或硬件未识别。
**解决**：
```bash
sudo apt install hailo-all   # 安装驱动+固件
sudo reboot
ls /dev/hailo0               # 确认设备存在
```

### 3. `HEF input shape (x,x,3) does not match configured imgsz=640`
**原因**：HEF 模型输入尺寸与配置不匹配。
**解决**：确认 `runtime.json` 中 `"imgsz": 640`，且 HEF 是 640×640 编译的。

### 4. `expected one Hailo NMS output, got [...]`
**原因**：HEF 输出层不是单个 NMS tensor。
**解决**：用 `hailortcli parse-hef` 检查输出结构。本项目 HEF 使用 Hailo NMS 后处理（alls 中 `nms_postprocess`），输出 1 个 NMS tensor。

### 5. `cannot open source: ...`
**原因**：视频文件路径错误或 RTSP 不可达。
**解决**：用绝对路径或确认文件存在。摄像头用 `--source 0`。

### 6. `cannot create output video`
**原因**：输出目录不可写或编解码器缺失。
**解决**：`mkdir -p outputs/`，确认安装了 ffmpeg。

### 7. Web 页面不显示视频
- 确认 `http://<PI_IP>:8000` 能访问（防火墙开放 8000 端口）
- 检查 `/api/status` 返回的 `error` 字段
- 首次启动需等几秒加载 HEF 到 Hailo

---

## PC 验证 vs 物理 Hailo 验证

### PC 验证（无 Hailo 硬件）

在开发 PC 上只能做静态和逻辑验证，**不能**运行真实 Hailo 推理：

```bash
# 语法和配置校验
python tools/check_deployment.py

# 无硬件单元测试
python -m pytest tests/test_runtime.py -v

# 配置验证（不加载 HailoRT）
python app/infer_video_hailo.py --check-config
python web_detection.py --check-config
```

> **PC 上不应声称 Hailo 推理已跑通。** `output/demo_yolov8n_pc.mp4` 是此前用
> Ultralytics 在 CPU 上的 YOLOv8n 验证结果，仅供模型精度参考，非 Hailo 推理。

### 物理 Hailo 验证（树莓派上）

```bash
# 1. 设备扫描
hailortcli scan
hailortcli fw-control identify

# 2. HEF 解析
hailortcli parse-hef models/hef/pothole_yolov8n_hailo8.hef

# 3. Python 探测
./scripts/probe.sh

# 4. 视频推理
./scripts/run_demo.sh

# 5. Web 预览
.venv/bin/python web_detection.py --source input/demo.mp4

# 6. 静态校验
.venv/bin/python tools/check_deployment.py
.venv/bin/python -m pytest tests/test_runtime.py -v
```

需记录：
- HEF 是否成功加载
- 输出 tensor / NMS 结构
- 实际 FPS
- 检测框是否正确
- 是否存在明显漏检或误检

---

## License

本项目仅用于教育和研究目的。
