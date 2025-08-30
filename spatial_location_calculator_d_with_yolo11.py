#!/bin/env python
# coding=utf-8
from __future__ import annotations

from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from spatial_location_calculator import (
    CALCULATION_ALGORITHMS,
    SpatialCalculatorBase,
    SpatialCalculatorConfig,
)
from utils.fps_handler import FPSHandler
from utils.image_processing import adjust_depth_frame, draw_rect, draw_text

spatial_config = SpatialCalculatorConfig(
    extended_disparity=False,
    subpixel=True,
    lr_check=True,
    calculation_algorithm="median",
    lower_threshold=0,
    upper_threshold=100_000,
)
# 对广 FOV 相机拍摄的图片进行去畸变时，变形会导致图片的某些部分被截断。
# 这导致了 FOV 的损失，而 FOV 是 WFOV 相机的主要卖点。
# 为了最大化 FOV，可以使用 alpha 参数 [0-1] 来缩放未扭曲的图像。
# https://docs.luxonis.com/software/perception/rgb-d/#RGB-D-Alignment%20using%20DepthAI-RGB-D%20on%20Wide%20FOV%20cameras-Alpha%20and%20maximizing%20FOV
ALPHA = 0

blob = Path(__file__).parent.joinpath("models", "yolov11n_openvino_2021.4_5shave.blob")
model = dai.OpenVINO.Blob(blob)
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

output_name, output_tenser = next(iter(model.networkOutputs.items()))
NUM_CLASSES = (
    output_tenser.dims[2] - 5
    if "yolov6" in output_name
    else output_tenser.dims[2] // 3 - 5
)

# fmt: off
LABEL_MAP = [
   "air_switch", "contactor", "hammer", "spanner", "pliers", "wire_tripper"
]

# fmt: on

# Weights to use when blending depthQueueData/rgb image (should equal 1.0)
RGB_WEIGHT = 0.4
DEPTH_WEIGHT = 0.6


def update_blend_weights(percent_rgb):
    """
    Update the rgb and depthQueueData weights used to blend depthQueueData/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global DEPTH_WEIGHT
    global RGB_WEIGHT
    RGB_WEIGHT = float(percent_rgb) / 100.0
    DEPTH_WEIGHT = 1.0 - RGB_WEIGHT


class SpatialCalculator(SpatialCalculatorBase):
    def __init__(
        self,
        device_id: str | None = None,
        roi_length: int = 1,
        spatial_config=spatial_config,
    ) -> None:
        self.spatial_calc_config_in_queue = None
        self.sync_group_queue = None
        self.first: list[bool] = [True, True, True]
        super().__init__(device_id, roi_length, spatial_config)

    def create_pipeline(self) -> dai.Pipeline:
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        center = pipeline.create(dai.node.Camera)
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)

        stereo = pipeline.create(dai.node.StereoDepth)
        spatial_location_calculator = pipeline.create(
            dai.node.SpatialLocationCalculator
        )
        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

        sync_node = pipeline.create(dai.node.Sync)
        xout_sync = pipeline.create(dai.node.XLinkOut)
        xin_spatial_calc_config = pipeline.create(dai.node.XLinkIn)

        xout_sync.setStreamName("syncGroup")
        xin_spatial_calc_config.setStreamName("spatialCalcConfig")

        # Properties
        center.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        # center.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        center.setFps(30)
        # center.setIspScale(2, 3)
        center.setPreviewSize(W, H)
        center.setSize(1280, 720)
        center.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        left.setFps(30)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        right.setFps(30)

        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        manip.setMaxOutputFrameSize(
            center.getPreviewWidth() * center.getPreviewHeight() * 3
        )

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        # LR-check is required for depthQueueData alignment
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.initialConfig.setLeftRightCheck(self.config.lr_check)
        stereo.initialConfig.setExtendedDisparity(self.config.extended_disparity)
        stereo.initialConfig.setSubpixel(self.config.subpixel)
        # stereo.setOutputSize(left.getResolutionWidth(), left.getResolutionHeight())
        if ALPHA is not None:
            center.setCalibrationAlpha(ALPHA)
            stereo.setAlphaScaling(ALPHA)

        # Network specific settings
        spatialDetectionNetwork.setBlob(model)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(NUM_CLASSES)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors([])
        spatialDetectionNetwork.setAnchorMasks({})
        spatialDetectionNetwork.setIouThreshold(0.3)

        # spatial specific parameters
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(self.config.lower_threshold)
        spatialDetectionNetwork.setDepthUpperThreshold(self.config.upper_threshold)
        spatialDetectionNetwork.setSpatialCalculationAlgorithm(
            CALCULATION_ALGORITHMS[self.config.calculation_algorithm]
        )

        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = self.config.lower_threshold
        config.depthThresholds.upperThreshold = self.config.upper_threshold
        config.calculationAlgorithm = CALCULATION_ALGORITHMS[
            self.config.calculation_algorithm
        ]

        for i in range(self.roi_length):
            top_left = dai.Point2f(0.1 + i * 0.1, 0.35)
            bottom_right = dai.Point2f(0.2 + i * 0.1, 0.45)
            self._add_roi(spatial_location_calculator, top_left, bottom_right)

        spatial_location_calculator.inputConfig.setWaitForMessage(False)

        # Linking
        sync_node.out.link(xout_sync.input)

        center.video.link(sync_node.inputs["color"])
        center.preview.link(manip.inputImage)
        manip.out.link(spatialDetectionNetwork.input)

        left.out.link(stereo.left)
        right.out.link(stereo.right)

        stereo.disparity.link(sync_node.inputs["depth"])
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(
            spatial_location_calculator.inputDepth
        )

        spatialDetectionNetwork.out.link(sync_node.inputs["detect_data"])

        spatial_location_calculator.out.link(sync_node.inputs["spatial_data"])

        xin_spatial_calc_config.out.link(spatial_location_calculator.inputConfig)

        return pipeline

    def create_output_queue(self):
        self.sync_group_queue = self.device.getOutputQueue(
            "syncGroup", maxSize=1, blocking=False
        )

    def create_input_queue(self):
        self.spatial_calc_config_in_queue = self.device.getInputQueue(
            "spatialCalcConfig"
        )

    def update_roi(self, key):
        return self._update_single_roi(key, 0, ord("w"), ord("a"), ord("s"), ord("d"))

    @staticmethod
    def frame_norm(frame, bbox):
        """
        nn data, being the bounding box locations, are in <0..1> range
        - they need to be normalized with frame width/height
        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def draw_detection(self, frame, detections):
        for detection in detections:
            bbox = self.frame_norm(
                frame,
                (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
            draw_text(
                frame,
                LABEL_MAP[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
            )
            draw_text(
                frame,
                f"{detection.confidence:.2%}",
                (bbox[0] + 10, bbox[1] + 35),
            )
            draw_rect(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 0, 255),
                (0, 0, 0),
            )
            if hasattr(detection, "boundingBoxMapping"):
                draw_text(
                    frame,
                    f"X: {int(detection.spatialCoordinates.x)} mm",
                    (bbox[0] + 10, bbox[1] + 50),
                )
                draw_text(
                    frame,
                    f"Y: {int(detection.spatialCoordinates.y)} mm",
                    (bbox[0] + 10, bbox[1] + 65),
                )
                draw_text(
                    frame,
                    f"Z: {int(detection.spatialCoordinates.z)} mm",
                    (bbox[0] + 10, bbox[1] + 80),
                )

    def run(self):
        with self.device:
            # Configure windows; trackbar adjusts blending ratio of rgb/depthQueueData
            rgb_window_name = "image"
            depth_window_name = "depthQueueData"
            blended_window_name = "rgb-depthQueueData"
            cv2.namedWindow(rgb_window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(blended_window_name, cv2.WINDOW_NORMAL)

            cv2.createTrackbar(
                "RGB Weight %",
                blended_window_name,
                int(RGB_WEIGHT * 100),
                100,
                update_blend_weights,
            )

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

                detect_packet = msg_group["detect_data"]  # type: dai.SpatialImgDetections | dai.ADatatype
                detections = detect_packet.detections
                for frame in [color_frame, depth_color_frame]:
                    self.draw_detection(frame, detections)

                blended = cv2.addWeighted(
                    color_frame, RGB_WEIGHT, depth_color_frame, DEPTH_WEIGHT, 0
                )

                if self.first[2]:
                    blended_shape = blended.shape
                    cv2.setMouseCallback(blended_window_name, self.on_mouse, blended_shape)
                    self.first[2] = False

                fps.draw_fps(color_frame, "image")
                fps.draw_fps(depth_color_frame, "depth")

                cv2.imshow(rgb_window_name, color_frame)
                cv2.imshow(depth_window_name, depth_color_frame)
                cv2.imshow(blended_window_name, blended)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if self.update_roi(key) or self.set_calculation_algorithm(key):
                    self.new_config = True

                if self.new_config:
                    self._update_spatial_calculator_config(
                        self.spatial_calc_config_in_queue
                    )


if __name__ == "__main__":
    with SpatialCalculator(
        device_id=None, roi_length=1, spatial_config=spatial_config
    ) as spatial_calculator:
        spatial_calculator.run()
