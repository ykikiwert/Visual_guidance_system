# coding=utf-8
from __future__ import annotations

import cv2
import numpy as np

BG_COLOR = (100, 100, 100)


def draw_text(
    frame: cv2.Mat,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = BG_COLOR,
    font_scale: float = 0.5,
    thickness: int = 1,
):
    """
    在图像上绘制文本

    Args:
        frame:要在其上绘制文本的图像
        text:要绘制的文本字符串
        org:文本的起始坐标
        color:文本颜色
        bg_color:文本背景颜色
        font_scale:字体缩放比例
        thickness:文本线条的粗细

    """
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bg_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_rect(
    frame: cv2.Mat,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = BG_COLOR,
    thickness: int = 1,
):
    """
    在图像上绘制矩形

    Args:
        frame:要在其上绘制矩形的图像
        top_left:矩形的左上角坐标
        bottom_right:矩形的右下角坐标
        color:矩形颜色
        bg_color:矩形背景颜色
        thickness:矩形线条的粗细

    """
    cv2.rectangle(frame, top_left, bottom_right, bg_color, thickness + 3, cv2.LINE_AA)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness, cv2.LINE_AA)


def draw_polylines(
    frame: cv2.Mat,
    pts: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = BG_COLOR,
    thickness: int = 1,
):
    """
    在图像上绘制矩形

    Args:
        frame:要在其上绘制矩形的图像
        pts:多边形的顶点
        color:矩形颜色
        bg_color:矩形背景颜色
        thickness:矩形线条的粗细

    """
    cv2.polylines(frame, pts, isClosed=True, color=bg_color, thickness=thickness + 3, lineType=cv2.LINE_AA)
    cv2.polylines(frame, pts, isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def adjust_depth_frame(depth_frame):
    """
    调整深度帧以适合显示

    Args:
        depth_frame(cv2.Mat|np.ndarray):要处理的深度帧

    Returns:
        调整后的深度帧
    """
    depth_downscaled = depth_frame[::4]
    min_depth = 0 if np.all(depth_downscaled == 0) else np.percentile(depth_downscaled[depth_downscaled != 0], 1)
    max_depth = np.percentile(depth_downscaled, 99)
    depth_frame_color = np.interp(depth_frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
    return cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)
