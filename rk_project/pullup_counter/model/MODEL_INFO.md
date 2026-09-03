# Model information

- File: `yolo11n-pose.rknn`
- Architecture: YOLO11n Pose
- Target platform: RK3576
- RKNN-Toolkit2: 2.3.2
- Precision: FP16 (not INT8)
- Input: RGB `uint8`, NHWC, `1×640×640×3`
- Internal normalization: mean `[0,0,0]`, std `[255,255,255]`
- Output: float tensor `1×56×8400`
- Output layout: `4 box + 1 person score + 17×(x,y,confidence)`
- SHA-256: `251B72CC1C3F4D2D0D5CEF3F5FB43FC38EC3B8C2AA3193613EB9C5A0CD68A82B`

The generic filename does not change model precision. This file is the FP16
model generated for RK3576. Replace it with a genuine INT8 build only after
representative calibration and pose/counting accuracy comparison.
