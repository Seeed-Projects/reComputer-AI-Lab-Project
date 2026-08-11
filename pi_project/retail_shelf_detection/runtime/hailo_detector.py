"""Synchronous HailoRT 4.x detector wrapper for Raspberry Pi 5."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .yolo_postprocess import letterbox_rgb, postprocess_auto

logger = logging.getLogger("runtime.hailo_detector")


class HailoDetector:
    # Both models are configured on one physical Hailo-8. Opening an independent
    # VDevice per model can make the second model fail with "device in use".
    _shared_device = None
    _shared_users = 0

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int,
        conf: float,
        iou: float,
        input_mode: str = "float32",
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.input_mode = input_mode.lower()
        self._printed_shapes = False
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"HEF model not found: {self.model_path}. Compile or copy the HEF first."
            )
        if self.input_mode not in {"float32", "uint8"}:
            raise ValueError("input_mode must be 'float32' or 'uint8'")
        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HEF,
                HailoSchedulingAlgorithm,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "pyHailoRT is unavailable. On Raspberry Pi OS install the matching "
                "hailo-all/python3-hailort packages; do not pip-install RKNN packages."
            ) from exc

        self._infer_type = FormatType.FLOAT32 if self.input_mode == "float32" else FormatType.UINT8
        if HailoDetector._shared_device is None:
            device_params = VDevice.create_params()
            device_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            HailoDetector._shared_device = VDevice(device_params)
        HailoDetector._shared_users += 1
        self._vdevice = HailoDetector._shared_device
        self._hef = HEF(str(self.model_path))
        params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        groups = self._vdevice.configure(self._hef, params)
        if len(groups) != 1:
            raise RuntimeError(f"expected one HEF network group, got {len(groups)}")
        self._network_group = groups[0]
        input_params = InputVStreamParams.make(
            self._network_group, quantized=self.input_mode == "uint8", format_type=self._infer_type
        )
        output_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self._pipeline_context = InferVStreams(self._network_group, input_params, output_params)
        self._pipeline = self._pipeline_context.__enter__()
        input_infos = self._hef.get_input_vstream_infos()
        if len(input_infos) != 1:
            raise RuntimeError(f"expected one model input, got {len(input_infos)}")
        self._input_name = input_infos[0].name
        logger.info("loaded HEF %s (input=%s, imgsz=%d)", self.model_path, self._input_name, self.imgsz)

    def predict(self, frame_bgr: np.ndarray) -> tuple[list[list[float]], list[float]]:
        height, width = frame_bgr.shape[:2]
        input_rgb = letterbox_rgb(frame_bgr, self.imgsz)
        if self.input_mode == "float32":
            # Hailo Model Zoo places the 0..255 -> model-range normalization in
            # the HEF, so non-quantized input also stays in the 0..255 range.
            input_rgb = input_rgb.astype(np.float32)
        batch = input_rgb[None, ...]
        # The shared VDevice scheduler activates the appropriate network group.
        outputs = self._pipeline.infer({self._input_name: batch})
        if not self._printed_shapes:
            logger.info(
                "%s output tensors: %s",
                self.model_path.name,
                {name: tuple(np.asarray(value).shape) for name, value in outputs.items()},
            )
            self._printed_shapes = True
        return postprocess_auto(outputs, height, width, self.conf, self.iou, self.imgsz)

    def release(self) -> None:
        pipeline_context = getattr(self, "_pipeline_context", None)
        if pipeline_context is not None:
            pipeline_context.__exit__(None, None, None)
            self._pipeline_context = None
        if getattr(self, "_vdevice", None) is not None:
            HailoDetector._shared_users = max(0, HailoDetector._shared_users - 1)
            if HailoDetector._shared_users == 0:
                device = HailoDetector._shared_device
                if device is not None and hasattr(device, "release"):
                    device.release()
                HailoDetector._shared_device = None
            self._vdevice = None

    def __enter__(self) -> "HailoDetector":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
