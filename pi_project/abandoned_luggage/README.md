# Raspberry Pi 5 · Hailo-8 Abandoned Luggage Detection

基于 **YOLO11m**（COCO-80）的遗留行李检测系统，从原本面向 **RK3576** 的 PC 项目移植到
**Raspberry Pi 5 + Hailo-8**。检测机场/车站/地铁中的遗留行李并触发报警，支持视频批量
推理和 Web 实时预览。

---

## 兼容性

| 组件 | 版本 |
|------|------|
| 主板 | Raspberry Pi 5 (aarch64) |
| 加速器 | Hailo-8 M.2 Key M 2280 (`/dev/hailo0`) |
| HailoRT | 4.23.0 |
| Python | 3.11 |
| OS | Raspberry Pi OS 64-bit Bookworm |

> HEF 面向 **Hailo-8**，不兼容 Hailo-8L。

---

## 项目结构

```
RaspberryPi5-Hailo8-abandoned-luggage/
├── app/
│   └── infer_video_hailo.py      # 视频推理入口
├── abandoned_monitor/            # 移植自原 RK3576 项目 abandoned_detection_v2.py
│   ├── processor.py              # 帧处理流水线（检测→ROI→跟踪→owner→遗弃判定→绘制）
│   └── tracker.py                # IoU 跟踪器（替代 ByteTrack，按类别分 ID 空间）
├── configs/
│   ├── runtime.json              # 运行时配置
│   └── device_target.json
├── conversion/                   # 模型转换（PT→ONNX→HEF），见 SOP
│   ├── calibration/              # 从演示视频抽取的 136 张校准图
│   ├── hailO_config/             # yolov11m NMS JSON + ALLS
│   ├── logs/                     # 编译日志
│   └── prepare_calibration.py
├── input/demo.mp4                # 演示视频（原项目 test2.mp4，ROI 匹配）
├── models/
│   ├── source_pt/yolo11m.pt
│   ├── onnx/yolo11m.onnx         # opset 11 静态 640，无 NMS
│   └── hef/yolov11m_abandoned_hailo8.hef   # 转换产物
├── runtime/
│   ├── hailo_detector.py         # HailoRT 推理（共享 VDevice + ROUND_ROBIN）
│   ├── yolo_postprocess.py       # 80 类 Hailo NMS 解码（含类别识别）
│   └── ultralytics_detector.py   # CPU 演示模式（PC 无 Hailo）
├── scripts/
│   ├── install_rpi5.sh / probe.sh / run_demo.sh / docker_run.sh
│   └── compile_yolov11m_hef.sh   # HEF 编译脚本（WSL 内执行）
├── tools/ check_deployment.py / probe_hailo.py
├── tests/ test_runtime.py        # 无硬件单元测试
├── web_detection.py              # FastAPI/MJPEG Web 预览
├── Dockerfile  .dockerignore  requirements*.txt  README.md
```

---

## 算法（忠实移植原项目）

1. **检测**：YOLO11m 检测 COCO-80；只保留 person(0)、backpack(24)、handbag(26)、suitcase(28)
2. **ROI 过滤**：仅处理矩形 ROI 内中心点的检测（默认 `[200,200,1100,800]`，对应 1920×1080）
3. **跟踪**：IoU 跟踪器按类别分配持久 ID（Hailo NMS 无跟踪 ID，替代原 ByteTrack）
4. **Owner 关联**：每个行李关联最近的人
5. **遗弃判定**：owner 距离 > `dist_threshold`(200px) **或** owner 消失，
   持续 > `time_threshold`(5s) → 报警
6. **静态持久化**：行李短暂丢失时保持最后位置最多 `bag_persistence_frames`(300帧)
7. **报警保持**：触发后保持 `alarm_hold`(5s) 防闪烁

### 框颜色

| 颜色 | 含义 |
|------|------|
| 黄色粗框 | ROI 区域 |
| 橙色 (255,200,0) | 人 (Person N) |
| 绿色 | 正常行李 |
| 黄色细框 | 静态持久化（当前帧未检测到，保持位置） |
| 红色 | 遗弃报警（ABANDONED!） |

---

## 模型信息

| 项目 | 值 |
|------|-----|
| 架构 | YOLO11m |
| 类别 | COCO-80（逻辑使用 person/backpack/handbag/suitcase） |
| 输入 | 640×640 RGB uint8 |
| 输出 | Hailo NMS（6 列或类别网格，含 class_id） |
| ONNX | `models/onnx/yolo11m.onnx`（opset 11，静态，无 NMS） |
| 转换流水线 | PT → ONNX → HAR → HEF，参考 `D:\Python Code\SOP_YOLO_PT_TO_ONNX_TO_HAILO8_HEF.md` |

---

## HEF 编译（在 WSL 中）

```bash
# 1. 导出 ONNX（Windows）已生成：models/onnx/yolo11m.onnx
# 2. 校准集已生成：conversion/calibration（136 张）

wsl.exe -d Ubuntu-22.04 -u seeed
source ~/.venvs/retail-hailo-dfc/bin/activate
bash "/mnt/d/Python Code/RaspberryPi5-Hailo8-abandoned-luggage/scripts/compile_yolov11m_hef.sh"
```

日志：`conversion/logs/yolov11m_abandoned.log`，产物：`models/hef/yolov11m_abandoned_hailo8.hef`

> 注意：m 级模型在纯 CPU 编译环境的分区搜索可能耗时较长（数小时），正常现象。

---

## 树莓派安装与运行

```bash
# 传输
scp -r RaspberryPi5-Hailo8-abandoned-luggage pi@<PI_IP>:~/

# 安装（aarch64 + Python 3.11 检查、本地 HailoRT wheel、不升级不联网）
cd ~/RaspberryPi5-Hailo8-abandoned-luggage
chmod +x scripts/*.sh
./scripts/install_rpi5.sh

# 设备检查
./scripts/probe.sh

# 视频推理（输出 + FPS + 遗弃计数）
./scripts/run_demo.sh
# 或 .venv/bin/python app/infer_video_hailo.py --config configs/runtime.json

# Web 预览
.venv/bin/python web_detection.py --source input/demo.mp4
# 浏览器 http://<PI_IP>:8000
```

### API

- `GET /` — Web 页面（MJPEG + 实时指标 + 颜色图例）
- `GET /api/video_feed` — MJPEG 流
- `GET /api/status` — JSON 状态（persons/bags/abandoned/alarm_frames）
- `GET /healthz` — 健康检查

---

## Docker

```bash
# 构建（arm64）
sudo docker build -f docker/hailo8/abandoned_luggage.dockerfile -t abandoned-luggage pi_project/abandoned_luggage 2>/dev/null || sudo docker build -t abandoned-luggage .
# 或直接在项目目录：sudo docker build -t abandoned-luggage .

./scripts/docker_run.sh web     # Web 预览
./scripts/docker_run.sh video   # 视频推理
```

`docker_run.sh` 映射 `/dev/hailo0`、`input/`、`output/`、`configs/` 和 `libhailort.so`。

---

## 配置参数（configs/runtime.json）

| 键 | 默认 | 说明 |
|----|------|------|
| `model.confidence` | 0.25 | 检测置信度阈值 |
| `model.iou` | 0.7 | NMS IoU |
| `roi.rect` | [200,200,1100,800] | ROI 矩形（1920×1080 画面） |
| `abandonment.dist_threshold` | 200 | owner 距离阈值（px） |
| `abandonment.time_threshold` | 5 | 离开持续时间（秒） |
| `abandonment.alarm_hold` | 5 | 报警保持（秒） |
| `abandonment.bag_persistence_frames` | 300 | 静态持久化帧数 |
| `tracking.match_iou` | 0.25 | 跟踪匹配 IoU |

命令行覆盖：`--source`、`--output`、`--confidence`、`--iou`、`--device {hailo,cpu}`。

---

## 测试（无硬件）

```bash
python -m pytest tests/test_runtime.py -v   # 19 项：letterbox/NMS解码/跟踪/几何/配置/视频/导入/语法
python tools/check_deployment.py            # 结构 + HEF SHA-256（编译完成后自动校验）
python app/infer_video_hailo.py --check-config
python web_detection.py --check-config
```

PC 上可用 `--device cpu` 完整跑通流水线（Ultralytics 版），验证遗弃逻辑，
但 **CPU 模式不代表 Hailo 推理已通过**——真实推理必须在树莓派上验证。

---

## 常见错误

| 错误 | 处理 |
|------|------|
| `pyHailoRT is unavailable` | 重跑 install_rpi5.sh 或本地装 wheel |
| `/dev/hailo0 not found` | `sudo apt install hailo-all && sudo reboot` |
| `expected one Hailo NMS output` | `hailortcli parse-hef` 检查输出结构 |
| `unsupported Hailo NMS output shape` | 确认 HEF 为 yolov8 系 NMS 输出 |
| 检测框全是 person 无 bag | 检查 ROI 是否覆盖区域、调整 confidence |
| 遗弃报警不触发 | 检查 `dist_threshold`/`time_threshold` 与画面尺度 |