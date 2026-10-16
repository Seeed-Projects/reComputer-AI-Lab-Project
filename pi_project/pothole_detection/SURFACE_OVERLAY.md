# 坑洞半透明覆盖

视频和 Web 共用 `PotholeProcessor`，现在默认叠加橙红色半透明表面和轮廓线。
YOLOv8n/Hailo 检测仍提供矩形框；轮廓由框内 GrabCut 和颜色对比估计，
不是分割模型掩膜，不能用于精确计算坑洞面积。复杂路面、反光或不准确的框
可能使覆盖区域偏离坑洞。无法可靠提取轮廓时只显示框。

`configs/runtime.json` 中的 `overlay`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| enabled | true | 开启表面覆盖，false 恢复仅框显示 |
| opacity | 0.34 | 透明度，0 不填充，1 完全遮盖 |
| color_bgr | [30,110,255] | OpenCV BGR 橙红色 |
| show_boxes | true | 保留细检测框与标签 |
| mask_size | 128 | 每个检测框内轮廓估计尺寸，32–256 |
| smoothing | 0.4 | 同一轨迹的归一化掩膜平滑权重，0–小于1 |

重叠检测只混合一次，防止颜色反复叠加。黄色 `pothole tracked` 代表短时
保留的历史检测框，没有当前帧直接检测，不填充表面。

树莓派/Hailo 输出视频：

```bash
python app/infer_video_hailo.py --output outputs/pothole_surface_hailo.mp4
```

Web 入口无需额外修改，会使用相同覆盖逻辑。GrabCut 会增加 CPU 开销，
树莓派实际 FPS 需要板端测量；可减小 mask_size 或关闭覆盖。

验证：

```bash
python -m pytest tests -q -p no:cacheprovider
```

包含合成图轮廓提取、重叠透明度、画面不被原地修改、越界框、跟踪保留框
不填充，以及配置范围验证。
