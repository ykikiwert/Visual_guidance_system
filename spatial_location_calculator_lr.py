#!/bin/env python
# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import depthai as dai
from spatial_location_calculator import (
    CALCULATION_ALGORITHMS,
    SpatialCalculatorBase,
    SpatialCalculatorConfig,
)
from utils.fps_handler import FPSHandler
from utils.image_processing import adjust_depth_frame


@dataclass
class SpatialCalculatorConfigLR(SpatialCalculatorConfig):
    """
    Spatial Calculator Configuration

    Attributes:
        CAM_GROUP (str): Camera group to use.
        extended_disparity (bool): Enable extended disparity to improve depth range.
                                Closer-in minimum depth, disparity range is doubled (from 95 to 190)
        subpixel (bool): Enable subpixel. Better accuracy for longer distance, fractional disparity 32-levels
        lr_check (bool): Enable left-right check. Better handling for occlusions
        calculation_algorithm (Literal): Algorithm to use for calculations.
        lower_threshold (int): Lower threshold in millimeters.
        upper_threshold (int): Upper threshold in millimeters.
    """

    CAM_GROUP: Literal["LC", "LR", "CR"] = "LR"  # LC, LR, CR


spatial_config = SpatialCalculatorConfigLR(
    CAM_GROUP="LR",
    extended_disparity=False,
    subpixel=True,
    lr_check=True,
    calculation_algorithm="mean",
    lower_threshold=0,
    upper_threshold=100_000,
)


class SpatialCalculatorLR(SpatialCalculatorBase):
    def __init__(
        self,
        device_id: str | None = None,
        roi_length: int = 1,
        spatial_config=spatial_config,
    ) -> None:
        self.spatial_calc_config_in_queue = None
        self.sync_group_queue = None
        self.first: list[bool] = [True, True]
        super().__init__(device_id, roi_length, spatial_config)

    def create_pipeline(self) -> dai.Pipeline:
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        left = pipeline.create(dai.node.ColorCamera)
        right = pipeline.create(dai.node.ColorCamera)

        stereo = pipeline.create(dai.node.StereoDepth)
        spatial_location_calculator = pipeline.create(dai.node.SpatialLocationCalculator)

        sync_node = pipeline.create(dai.node.Sync)
        xout_sync = pipeline.create(dai.node.XLinkOut)
        xin_spatial_calc_config = pipeline.create(dai.node.XLinkIn)

        xout_sync.setStreamName("syncGroup")
        xin_spatial_calc_config.setStreamName("spatialCalcConfig")

        # Properties
        left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        left.setIspScale(1, 3)
        left.setFps(30)
        right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        right.setIspScale(1, 3)
        right.setFps(30)

        if self.config.CAM_GROUP == "LC":
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        if self.config.CAM_GROUP == "LR":
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        if self.config.CAM_GROUP == "RC":
            left.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        # LR-check is required for depthQueueData alignment
        stereo.setDepthAlign(right.getBoardSocket())
        stereo.initialConfig.setLeftRightCheck(self.config.lr_check)
        stereo.initialConfig.setExtendedDisparity(self.config.extended_disparity)
        stereo.initialConfig.setSubpixel(self.config.subpixel)

        for i in range(self.roi_length):
            top_left = dai.Point2f(0.1 + i * 0.1, 0.35)
            bottom_right = dai.Point2f(0.2 + i * 0.1, 0.45)
            self._add_roi(spatial_location_calculator, top_left, bottom_right)

        spatial_location_calculator.inputConfig.setWaitForMessage(False)

        # Linking
        sync_node.out.link(xout_sync.input)

        stereo.syncedRight.link(sync_node.inputs["color"])

        left.isp.link(stereo.left)
        right.isp.link(stereo.right)

        stereo.disparity.link(sync_node.inputs["depth"])
        stereo.depth.link(spatial_location_calculator.inputDepth)

        spatial_location_calculator.out.link(sync_node.inputs["spatial_data"])

        xin_spatial_calc_config.out.link(spatial_location_calculator.inputConfig)

        return pipeline

    def create_output_queue(self):
        self.sync_group_queue = self.device.getOutputQueue("syncGroup", maxSize=1, blocking=False)

    def create_input_queue(self):
        self.spatial_calc_config_in_queue = self.device.getInputQueue("spatialCalcConfig")

    def update_roi(self, key):
        return self._update_single_roi(key, 0, ord("w"), ord("a"), ord("s"), ord("d"))

    def run(self):
        with self.device:
            # Configure windows; trackbar adjusts blending ratio of rgb/depthQueueData
            rgb_window_name = "image"
            depth_window_name = "depthQueueData"
            cv2.namedWindow(rgb_window_name)
            cv2.namedWindow(depth_window_name)

            fps = FPSHandler()
            while True:
                msg_group = self.sync_group_queue.get()  # type: dai.MessageGroup | dai.ADatatype
                frame_packet = msg_group["color"]  # type: dai.ImgFrame | dai.ADatatype
                fps.tick("image")
                color_frame = frame_packet.getCvFrame()
                if self.first[0]:
                    color_shape = color_frame.shape
                    cv2.setMouseCallback(rgb_window_name, self.on_mouse, color_shape)
                    self.first[0] = False

                depth_packet = msg_group["depth"]  # type: dai.ImgFrame | dai.ADatatype
                fps.tick("depth")
                depth_frame = depth_packet.getFrame()
                depth_color_frame = adjust_depth_frame(depth_frame)
                if self.first[1]:
                    depth_shape = depth_frame.shape
                    cv2.setMouseCallback(depth_window_name, self.on_mouse, depth_shape)
                    self.first[1] = False

                spatial_packet = msg_group["spatial_data"]  # type: dai.SpatialLocationCalculatorData | dai.ADatatype
                spatial_data = spatial_packet.getSpatialLocations()
                for depth_data in spatial_data:
                    for frame in [color_frame, depth_color_frame]:
                        self.draw_spatial_info(frame, depth_data, (255, 255, 2))

                fps.draw_fps(color_frame, "image")
                fps.draw_fps(depth_color_frame, "depth")
                cv2.imshow(rgb_window_name, color_frame)
                cv2.imshow(depth_window_name, depth_color_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if self.update_roi(key) or self.set_calculation_algorithm(key):
                    self.new_config = True

                if self.new_config:
                    self._update_spatial_calculator_config(self.spatial_calc_config_in_queue)


if __name__ == "__main__":
    with SpatialCalculatorLR(device_id=None, roi_length=1, spatial_config=spatial_config) as spatial_calculator:
        spatial_calculator.run()
