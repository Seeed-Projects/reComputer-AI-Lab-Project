"""Synchronous HailoRT 4.x detector wrapper for Raspberry Pi 5."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from .yolo_postprocess import letterbox_rgb, postprocess_auto

logger = logging.getLogger("runtime.hailo_detector")


class HailoDetector:
    def __init__(self, model_path: str | Path, imgsz: int, conf: float, iou: float,
                 input_mode: str = "uint8") -> None:
        self.model_path = Path(model_path)
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.input_mode = input_mode.lower()
        self._printed_shapes = False
        if not self.model_path.exists():
            raise FileNotFoundError(f"HEF model not found: {self.model_path}")
        if self.input_mode not in {"float32", "uint8"}:
            raise ValueError("input_mode must be float32 or uint8")
        try:
            from hailo_platform import (ConfigureParams, FormatType, HEF, HailoStreamInterface,
                InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)
        except ImportError as exc:
            raise RuntimeError("pyHailoRT is unavailable; install the matching HailoRT 4.23 package") from exc

        self._format = FormatType.FLOAT32 if self.input_mode == "float32" else FormatType.UINT8
        self._vdevice = VDevice()
        self._hef = HEF(str(self.model_path))
        params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        groups = self._vdevice.configure(self._hef, params)
        if len(groups) != 1:
            raise RuntimeError(f"expected one HEF network group, got {len(groups)}")
        self._network_group = groups[0]
        input_params = InputVStreamParams.make(
            self._network_group, quantized=self.input_mode == "uint8", format_type=self._format)
        output_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32)
        self._context = InferVStreams(self._network_group, input_params, output_params)
        self._pipeline = self._context.__enter__()
        input_infos = self._hef.get_input_vstream_infos()
        if len(input_infos) != 1:
            raise RuntimeError(f"expected one input, got {len(input_infos)}")
        self._input_name = input_infos[0].name
        input_shape = tuple(int(value) for value in input_infos[0].shape)
        if input_shape not in {(self.imgsz, self.imgsz, 3), (3, self.imgsz, self.imgsz)}:
            raise RuntimeError(
                f"HEF input shape {input_shape} does not match configured imgsz={self.imgsz}"
            )
        output_infos = self._hef.get_output_vstream_infos()
        if len(output_infos) != 1 or "nms" not in output_infos[0].name.lower():
            raise RuntimeError(
                "expected one Hailo NMS output, got "
                + str([(item.name, tuple(item.shape)) for item in output_infos])
            )
        logger.info(
            "loaded %s input=%s shape=%s output=%s",
            self.model_path.name, self._input_name, input_shape, output_infos[0].name,
        )

    def predict(self, frame_bgr: np.ndarray):
        height, width = frame_bgr.shape[:2]
        image = letterbox_rgb(frame_bgr, self.imgsz)
        if self.input_mode == "float32":
            image = image.astype(np.float32)
        outputs = self._pipeline.infer({self._input_name: image[None, ...]})
        if not self._printed_shapes:
            logger.info("output tensors: %s", {k: tuple(np.asarray(v).shape) for k, v in outputs.items()})
            self._printed_shapes = True
        return postprocess_auto(outputs, height, width, self.conf, self.iou, self.imgsz)

    def release(self) -> None:
        if getattr(self, "_context", None) is not None:
            self._context.__exit__(None, None, None)
            self._context = None
        if getattr(self, "_vdevice", None) is not None:
            if hasattr(self._vdevice, "release"):
                self._vdevice.release()
            self._vdevice = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
