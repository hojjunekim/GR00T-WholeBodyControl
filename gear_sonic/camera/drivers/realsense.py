"""Intel RealSense camera driver.

Requires the ``pyrealsense2`` SDK — install with::

    pip install pyrealsense2

See https://github.com/IntelRealSense/librealsense for hardware-specific instructions.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyrealsense2 as rs

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class RealSenseConfig:
    """Configuration for the RealSense camera."""

    depth_image_dim: tuple[int, int] = (640, 480)
    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    # Auto-exposure controls (applied to the COLOR sensor after pipeline.start).
    # auto_exposure=False uses manual_exposure_us (microseconds) instead.
    auto_exposure: bool = True
    manual_exposure_us: int = 8500


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for Intel RealSense depth cameras."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = RealSenseConfig(),
        id: int = 0,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")

        for device in devices:
            print(f"Device: {device.get_info(rs.camera_info.name)}")
            print(f"    Serial number: {device.get_info(rs.camera_info.serial_number)}")
            print(f"    Firmware version: {device.get_info(rs.camera_info.firmware_version)}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        devices = sorted(devices, key=lambda x: x.get_info(rs.camera_info.serial_number))
        self.config.enable_device(devices[id].get_info(rs.camera_info.serial_number))

        try:
            self.config.enable_stream(
                rs.stream.color,
                config.color_image_dim[0],
                config.color_image_dim[1],
                rs.format.rgb8,
                config.fps,
            )
            self.config.enable_stream(
                rs.stream.depth,
                config.depth_image_dim[0],
                config.depth_image_dim[1],
                rs.format.z16,
                config.fps,
            )
            profile = self.pipeline.start(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")

        # Stash the COLOR sensor so set_exposure() can flip AE / manual mode
        # on the live pipeline (pyrealsense2 supports option writes while
        # streaming — no need to restart). On D435i, sensor 0 is depth and
        # sensor 1 is RGB; first_color_sensor() finds it regardless of order.
        # Stash the pipeline profile so get_intrinsics() can query the per-stream
        # intrinsics/extrinsics later without restarting the device.
        self._profile = profile

        try:
            self._color_sensor = profile.get_device().first_color_sensor()
        except Exception as e:
            self._color_sensor = None
            print(f"[RealSense] WARNING: no color sensor found: {e}")

        self.set_exposure(config.auto_exposure, config.manual_exposure_us)

        self._realsense_config = config
        self._run_as_server = run_as_server
        self.mount_position = mount_position
        if self._run_as_server:
            self.start_server(port)
        print(
            f"Done initializing RealSense sensor: "
            f"{devices[id].get_info(rs.camera_info.serial_number)}"
        )

    def set_exposure(self, auto: bool, manual_us: int | None = None) -> str:
        """Toggle color-sensor auto-exposure on/off at runtime.

        pyrealsense2 accepts option writes on a live, streaming sensor, so this
        works without restarting the pipeline. Returns a status string suitable
        for logging back to the caller.
        """
        if self._color_sensor is None:
            return "no color sensor available"
        try:
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto else 0)
            if not auto and manual_us is not None:
                self._color_sensor.set_option(rs.option.exposure, float(manual_us))
            msg = f"AE={'ON' if auto else 'OFF'}"
            if not auto and manual_us is not None:
                msg += f" exposure={manual_us}us"
            print(f"[RealSense] {msg}")
            return msg
        except Exception as e:
            err = f"set_exposure failed: {e}"
            print(f"[RealSense] {err}")
            return err

    def set_white_balance(self, auto: bool, kelvin: int | None = None) -> str:
        """Toggle color-sensor auto-white-balance and/or set manual white balance."""
        if self._color_sensor is None:
            return "no color sensor available"
        try:
            self._color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if auto else 0)
            if not auto and kelvin is not None:
                self._color_sensor.set_option(rs.option.white_balance, float(kelvin))
            msg = f"AWB={'ON' if auto else 'OFF'}"
            if not auto and kelvin is not None:
                msg += f" white_balance={kelvin}K"
            print(f"[RealSense] {msg}")
            return msg
        except Exception as e:
            err = f"set_white_balance failed: {e}"
            print(f"[RealSense] {err}")
            return err

    def get_intrinsics(self) -> dict:
        """Return per-stream intrinsics and the depth->color extrinsics.

        Intrinsics are tied to the stream resolution configured at
        pipeline.start() — change RealSenseConfig.{color,depth}_image_dim
        and these values change too. fx/fy/cx/cy are in pixels; distortion
        coeffs follow the librealsense `model` convention.
        """
        if self._profile is None:
            return {"error": "no pipeline profile available"}
        try:
            color = (
                self._profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            depth = (
                self._profile.get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            extr = (
                self._profile.get_stream(rs.stream.depth)
                .get_extrinsics_to(self._profile.get_stream(rs.stream.color))
            )

            def _pack(i):
                return {
                    "width":  int(i.width),
                    "height": int(i.height),
                    "fx":     float(i.fx),
                    "fy":     float(i.fy),
                    "cx":     float(i.ppx),
                    "cy":     float(i.ppy),
                    "model":  str(i.model),
                    "coeffs": [float(c) for c in i.coeffs],
                }

            return {
                "color": _pack(color),
                "depth": _pack(depth),
                "depth_to_color": {
                    "rotation":    [float(r) for r in extr.rotation],   # 3x3 row-major
                    "translation": [float(t) for t in extr.translation],  # meters
                },
            }
        except Exception as e:
            return {"error": f"get_intrinsics failed: {e}"}

    def get_status(self) -> dict:
        """Read back the current color-sensor option values (for verification)."""
        if self._color_sensor is None:
            return {"error": "no color sensor"}
        try:
            return {
                "auto_exposure": int(self._color_sensor.get_option(rs.option.enable_auto_exposure)),
                "exposure_us": float(self._color_sensor.get_option(rs.option.exposure)),
                "gain": float(self._color_sensor.get_option(rs.option.gain)),
                "auto_white_balance": int(self._color_sensor.get_option(rs.option.enable_auto_white_balance)),
                "white_balance_k": float(self._color_sensor.get_option(rs.option.white_balance)),
            }
        except Exception as e:
            return {"error": str(e)}

    def read(self) -> dict[str, Any] | None:
        try:
            frames = self.pipeline.wait_for_frames()
        except Exception as e:
            print(f"ERROR! Failed to wait for frames: {e}")
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("WARNING! No color or depth frame")
            return None

        try:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        except Exception as e:
            print(f"ERROR! Failed to convert frames to numpy arrays: {e}")
            return None

        if color_image.size == 0 or depth_image.size == 0:
            print("WARNING! Empty color or depth image")
            return None

        current_time = time.time()
        timestamps = {
            self.mount_position: current_time,
            f"{self.mount_position}_depth": current_time,
        }
        images = {
            self.mount_position: color_image,
            f"{self.mount_position}_depth": depth_image,
        }
        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self._realsense_config.color_image_dim[1],
                        self._realsense_config.color_image_dim[0],
                        3,
                    ),
                    dtype=np.uint8,
                ),
                "depth_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self._realsense_config.depth_image_dim[1],
                        self._realsense_config.depth_image_dim[0],
                        1,
                    ),
                    dtype=np.uint16,
                ),
            }
        )

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.pipeline.stop()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
