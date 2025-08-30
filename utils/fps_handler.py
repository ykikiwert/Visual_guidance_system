# coding=utf-8
from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from utils.image_processing import draw_text

if TYPE_CHECKING:
    import numpy as np


class FPSHandler:
    """
    Class that handles all FPS-related operations.

    Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on its FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    def __init__(self, max_ticks: int = 100) -> None:
        """
        Constructor that initializes the class with a video file object and a maximum ticks amount for FPS calculation

        Args:
            max_ticks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._ticks: dict[str, deque[float]] = {}

        if max_ticks < 2:  # noqa: PLR2004
            msg = f"Provided max_ticks value must be 2 or higher (supplied: {max_ticks})"
            raise ValueError(msg)

        self._maxTicks = max_ticks

    def tick(self, name: str) -> None:
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tick_fps(self, name: str) -> float:
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def print_status(self):
        """Prints total FPS for all names stored in :func:`tick` calls"""
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tick_fps(name):.1f}")

    def draw_fps(self, frame: np.ndarray, name: str):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        draw_text(
            frame,
            frame_fps,
            (5, 15),
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )

        if "nn" in self._ticks:
            draw_text(
                frame,
                f"NN FPS:  {round(self.tick_fps('nn'), 1)}",
                (5, 30),
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
            )
