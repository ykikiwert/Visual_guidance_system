#!/bin/env python
# coding=utf-8
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import cv2
import depthai as dai
import numpy as np
from utils.device_manager import get_device_info
from utils.image_processing import draw_rect, draw_text

CALCULATION_ALGORITHMS = {
    "mean": dai.SpatialLocationCalculatorAlgorithm.MEAN,
    "min": dai.SpatialLocationCalculatorAlgorithm.MIN,
    "max": dai.SpatialLocationCalculatorAlgorithm.MAX,
    "mode": dai.SpatialLocationCalculatorAlgorithm.MODE,
    "median": dai.SpatialLocationCalculatorAlgorithm.MEDIAN,
}

STEP_SIZE = 0.05


@dataclass
class SpatialCalculatorConfig:
    """
    Spatial Calculator Configuration

    Attributes:
        extended_disparity (bool): Enable extended disparity to improve depth range.
                                Closer-in minimum depth, disparity range is doubled (from 95 to 190)
        subpixel (bool): Enable subpixel. Better accuracy for longer distance, fractional disparity 32-levels
        lr_check (bool): Enable left-right check. Better handling for occlusions
        calculation_algorithm (Literal): Algorithm to use for calculations.
        lower_threshold (int): Lower threshold in millimeters.
        upper_threshold (int): Upper threshold in millimeters.
    """

    extended_disparity: bool = False
    subpixel: bool = False
    lr_check: bool = True

    calculation_algorithm: Literal["mean", "median", "max", "min", "mode"] = "mean"
    lower_threshold: int = field(default=0, metadata={"check_bounds": True})
    upper_threshold: int = field(default=100_000, metadata={"check_bounds": True})

    def __post_init__(self):
        if self.lower_threshold < 0:
            msg = "Lower threshold must be non-negative."
            raise ValueError(msg)
        if self.upper_threshold <= 0:
            msg = "Upper threshold must be positive."
            raise ValueError(msg)
        if self.lower_threshold >= self.upper_threshold:
            msg = "Lower threshold must be less than upper threshold."
            raise ValueError(msg)

        if self.calculation_algorithm not in CALCULATION_ALGORITHMS:
            msg = f"Invalid calculation algorithm '{self.calculation_algorithm}'. Must be one of {list(CALCULATION_ALGORITHMS.keys())}."
            raise ValueError(msg)


class SpatialCalculatorBase(ABC):
    def __init__(
        self,
        device_id: str | None = None,
        roi_length: int = 1,
        # spatial_config: Callable[[], SpatialCalculatorConfig] = SpatialCalculatorConfig,
        spatial_config: SpatialCalculatorConfig = SpatialCalculatorConfig(),
    ) -> None:
        self.config = spatial_config
        self.roi_length = roi_length
        self.roi_list: deque[dai.Rect] = deque(maxlen=roi_length)
        self.new_config = False
        self.selecting = False
        self.start_point: list[float, float] | None = None
        self.end_point: list[float, float] | None = None

        self._device_info = get_device_info(device_id)
        self._pipeline = self.create_pipeline()
        self.device = dai.Device(self._pipeline, self._device_info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
        if self.device.getIrDrivers():
            self.device.setIrLaserDotProjectorIntensity(0.5)
        self.create_queue()

    def close(self):
        if not self.device.isClosed():
            self.device.close()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @abstractmethod
    def create_pipeline(self) -> dai.Pipeline:
        pass

    @abstractmethod
    def create_output_queue(self):
        pass

    @abstractmethod
    def create_input_queue(self):
        pass

    def create_queue(self):
        self.create_output_queue()
        self.create_input_queue()

    def _add_roi(self, spatial_location_calculator, top_left, bottom_right):
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = self.config.lower_threshold
        config.depthThresholds.upperThreshold = self.config.upper_threshold
        config.calculationAlgorithm = CALCULATION_ALGORITHMS[self.config.calculation_algorithm]
        roi = dai.Rect(top_left, bottom_right)
        config.roi = roi
        spatial_location_calculator.initialConfig.addROI(config)
        self.roi_list.append(roi)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x / param[1], y / param[0])
            self.end_point = None
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (
                max(min(x / param[1], 1.0), 0.0),
                max(min(y / param[0], 1.0), 0.0),
            )
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (
                max(min(x / param[1], 1.0), 0.0),
                max(min(y / param[0], 1.0), 0.0),
            )
            if self.start_point and self.end_point:
                top_left, bottom_right = (
                    dai.Point2f(
                        min(self.start_point[0], self.end_point[0]),
                        min(self.start_point[1], self.end_point[1]),
                    ),
                    dai.Point2f(
                        max(self.start_point[0], self.end_point[0]),
                        max(self.start_point[1], self.end_point[1]),
                    ),
                )
                self.roi_list.append(dai.Rect(top_left, bottom_right))
                self.new_config = True
                self.start_point, self.end_point = None, None

    @abstractmethod
    def update_roi(self, key):
        pass

    def _update_single_roi(self, key, index, up, left, down, right):
        if not self.roi_list[index]:
            return False
        roi = self.roi_list[index]
        top_left = roi.topLeft()
        bottom_right = roi.bottomRight()
        if key == up and top_left.y - STEP_SIZE >= 0:
            top_left.y -= STEP_SIZE
            bottom_right.y -= STEP_SIZE
        elif key == left and top_left.x - STEP_SIZE >= 0:
            top_left.x -= STEP_SIZE
            bottom_right.x -= STEP_SIZE
        elif key == down and bottom_right.y + STEP_SIZE <= 1:
            top_left.y += STEP_SIZE
            bottom_right.y += STEP_SIZE
        elif key == right and bottom_right.x + STEP_SIZE <= 1:
            top_left.x += STEP_SIZE
            bottom_right.x += STEP_SIZE
        else:
            return False
        self.roi_list[index] = dai.Rect(top_left, bottom_right)
        return True

    def set_calculation_algorithm(self, key):
        algorithms = {
            ord("1"): "mean",
            ord("2"): "min",
            ord("3"): "max",
            ord("4"): "mode",
            ord("5"): "median",
        }
        if key in algorithms:
            calculation_algorithm = algorithms[key]
            print(f"Switching calculation algorithm to {calculation_algorithm}!")
            self.config.calculation_algorithm = calculation_algorithm
            return True
        return False

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def draw_spatial_info(frame, depth_data, color):
        raw_roi = depth_data.config.roi
        roi = raw_roi.denormalize(width=frame.shape[1], height=frame.shape[0])
        x_min, y_min, x_max, y_max = np.rint([
            roi.topLeft().x,
            roi.topLeft().y,
            roi.bottomRight().x,
            roi.bottomRight().y,
        ]).astype(int)  # .tolist()

        draw_rect(frame, (x_min, y_min), (x_max, y_max), color)
        if hasattr(depth_data, "spatialCoordinates"):
            coordinates: dai.Point3f = depth_data.spatialCoordinates
            draw_text(frame, f"X: {coordinates.x:4.4f} cm", (x_min + 10, y_min + 20), color)
            draw_text(frame, f"Y: {coordinates.y:4.4f} cm", (x_min + 10, y_min + 35), color)
            draw_text(frame, f"Z: {coordinates.z:4.4f} cm", (x_min + 10, y_min + 50), color)

    def _update_spatial_calculator_config(self, spatial_calc_config_in_queue):
        cfg = dai.SpatialLocationCalculatorConfig()

        for rect in self.roi_list:
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = self.config.lower_threshold
            config.depthThresholds.upperThreshold = self.config.upper_threshold
            config.calculationAlgorithm = CALCULATION_ALGORITHMS[self.config.calculation_algorithm]
            config.roi = rect
            # print(f"Adding ROI: {rect.x, rect.y, rect.width, rect.height}")
            cfg.addROI(config)
        spatial_calc_config_in_queue.send(cfg)
        self.new_config = False
