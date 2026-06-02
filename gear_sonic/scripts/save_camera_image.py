"""Connect to a camera ZMQ server, read one ego_view frame, save to disk.

Usage:
    python gear_sonic/scripts/save_camera_image.py \\
        --camera-host 192.168.123.164 \\
        --camera-port 5555 \\
        --out /tmp/camera_frame.jpg
"""
import argparse
import time
import cv2

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera-host", default="192.168.123.164")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--out", default="/tmp/camera_frame.jpg")
    p.add_argument("--stream", default="ego_view",
                   help="image stream name (e.g. ego_view, ego_view_depth)")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="seconds to wait for a frame")
    args = p.parse_args()

    print(f"Connecting to {args.camera_host}:{args.camera_port} ...")
    client = ComposedCameraClientSensor(server_ip=args.camera_host, port=args.camera_port)

    t0 = time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        msg = client.read()
        if msg is not None and args.stream in msg.get("images", {}):
            img_rgb = msg["images"][args.stream]
            # uint8 HxWx3 RGB → cv2 expects BGR
            cv2.imwrite(args.out, img_rgb[:, :, ::-1])
            print(f"Saved {args.stream} {img_rgb.shape} {img_rgb.dtype} → {args.out}")
            return
        time.sleep(0.05)

    print(f"Timed out after {args.timeout}s — no '{args.stream}' frame received")


if __name__ == "__main__":
    main()
