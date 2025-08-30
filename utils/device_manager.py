#!/usr/bin/env python3
# coding: utf-8
# device_manager.py
from __future__ import annotations

import depthai as dai


class DeviceManager:
    """管理设备的工具类，用于获取和选择 DepthAI 设备"""

    @classmethod
    def get_available_devices(cls, *, debug: bool = False, exclude_poe: bool = True) -> list[dai.DeviceInfo]:
        """获取所有可用设备，并根据 PoE 过滤

        Args:
        - debug (bool): 是否启用调试模式。
        - exclude_poe (bool): 是否排除 PoE 设备。

        Returns:
            List[dai.DeviceInfo]: Returns 可用设备的列表。
        """
        device_infos = dai.XLinkConnection.getAllConnectedDevices() if debug else dai.Device.getAllAvailableDevices()

        if not exclude_poe:
            device_infos = [
                device_info for device_info in device_infos if device_info.protocol != dai.XLinkProtocol.X_LINK_TCP_IP
            ]

        return device_infos

    @classmethod
    def list_devices(cls, device_infos: list[dai.DeviceInfo]) -> None:
        """打印可用设备信息

        Args:
            device_infos (List[dai.DeviceInfo]): 设备信息列表。
        """
        if not device_infos:
            print("No available devices found.")
            return

        print("Available devices:")
        for i, device_info in enumerate(device_infos):
            print(f"[{i}] {device_info.name} {device_info.getMxId()} [{device_info.state.name}]")

    @classmethod
    def choose_device(
        cls, device_id: str | None, device_infos: list[dai.DeviceInfo], *, force: bool = False
    ) -> dai.DeviceInfo:
        """根据用户输入选择设备

        Args:
            device_id (str): 用户输入的设备 ID。
            device_infos (List[dai.DeviceInfo]): 设备信息列表。
            force (bool): 是否强制选择设备，即使设备 ID 不存在。

        Returns:
            dai.DeviceInfo: 选定的设备信息。

        Raises:
            SystemExit: 当用户选择退出时。
        """
        cls.list_devices(device_infos)

        if device_id == "list":
            raise SystemExit(0)

        # 查找匹配的设备
        selected_device = next(
            (info for info in device_infos if info.getMxId() == device_id or info.name == device_id), None
        )
        if selected_device:
            return selected_device
        if force and device_id is not None:
            return dai.DeviceInfo(device_id)

        # 如果设备 ID 未提供，决定是否只返回一台设备或提示选择
        if len(device_infos) == 1:
            return device_infos[0]

        return cls.prompt_user_for_device(device_infos)

    @classmethod
    def prompt_user_for_device(cls, device_infos: list[dai.DeviceInfo]) -> dai.DeviceInfo:
        """提示用户选择设备

        Args:
            device_infos (List[dai.DeviceInfo]): 设备信息列表。

        Returns:
            dai.DeviceInfo: 选定的设备信息。
        """
        while True:
            val = input("Which DepthAI Device you want to use (input the index or 'exit'/'q' to quit): ")
            if val.lower() in {"exit", "q"}:
                print("Exiting the program.")
                raise SystemExit(0)

            if val.isdigit() and int(val) in range(len(device_infos)):
                return device_infos[int(val)]
            print("Invalid input. Please try again.")


def get_device_info(
    device_id: str | None = None, *, force: bool = False, debug: bool = False, exclude_poe: bool = True
) -> dai.DeviceInfo:
    """
    获取设备信息，根据提供的设备 ID、调试模式和 PoE 选项。

    Args:
        device_id (Optional[str]): 设备 ID，如果为 "list" 则列出所有设备。
        force (bool): 是否强制选择设备，即使设备 ID 不存在。
        debug (bool): 是否以调试模式获取所有连接的设备。
        exclude_poe (bool): 是否排除 PoE 设备。

    Returns:
        dai.DeviceInfo: 请求的设备信息。

    Raises:
        RuntimeError: 如果没有找到设备或提供的设备 ID 无效。
    """
    device_infos = DeviceManager.get_available_devices(debug=debug, exclude_poe=exclude_poe)

    if len(device_infos) == 0 and device_id is None:
        msg = "No DepthAI device found!"
        raise RuntimeError(msg)

    return DeviceManager.choose_device(device_id, device_infos, force=force)
