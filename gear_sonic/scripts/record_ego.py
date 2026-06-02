"""Headless ego-view recorder. Pipes raw frames from the ZMQ camera server
into an `ffmpeg` H.264 encoder until Ctrl+C, producing a portable mp4 that
plays in any standard player. Run in a separate terminal while a policy
deploy is running — both can read the same ZMQ camera stream concurrently.

Virtual env: same as save_camera_image.py / run_camera_viewer.py.
    source .venv_data_collection/bin/activate

Usage (defaults match the real-robot SONIC camera):
    python gear_sonic/scripts/record_ego.py \\
        --camera-host 192.168.123.164 --camera-port 5555

Output:
    camera_recordings/ego_<YYYYMMDD_HHMMSS>.mp4
"""
import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera-host", default="192.168.123.164")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--stream", default="ego_view")
    p.add_argument("--out-dir", default="camera_recordings")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--crf", type=int, default=23,
                   help="ffmpeg H.264 quality (lower = better, 18-28 typical).")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        print("[record_ego] ffmpeg not found on PATH. Install via: sudo apt install ffmpeg",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ego_{ts}.mp4"

    print(f"[record_ego] Connecting to {args.camera_host}:{args.camera_port} ...")
    client = ComposedCameraClientSensor(server_ip=args.camera_host, port=args.camera_port)

    # Wait for first frame so we know image size.
    deadline = time.monotonic() + args.connect_timeout
    first = None
    while time.monotonic() < deadline:
        msg = client.read()
        if msg is not None and args.stream in msg.get("images", {}):
            first = msg["images"][args.stream]
            break
        time.sleep(0.02)
    if first is None:
        print(f"[record_ego] No '{args.stream}' frame in {args.connect_timeout}s — is the camera server up?",
              file=sys.stderr)
        sys.exit(1)

    h, w = first.shape[:2]

    # Spawn ffmpeg: read raw RGB24 frames from stdin, write H.264 mp4.
    # `-fflags +genpts` + the input fps tell ffmpeg the wall-clock framerate
    # so playback is real-time even if our write loop is bursty.
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", str(args.crf),
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"[record_ego] Writing → {out_path}  ({w}x{h} @ {args.fps:.1f} fps, H.264 crf={args.crf}).  Ctrl+C to stop.")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    stop = False
    def _sigint(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    frames = 0
    t_start = time.monotonic()
    last_rgb = None

    try:
        # Seed the file with the buffered first frame so the recording starts
        # exactly when we connect, not after the first ZMQ poll.
        proc.stdin.write(first.tobytes())
        frames += 1

        while not stop:
            msg = client.read()
            if msg is None or args.stream not in msg.get("images", {}):
                time.sleep(0.005)
                continue
            rgb = msg["images"][args.stream]
            if rgb is last_rgb:
                time.sleep(0.002)
                continue
            last_rgb = rgb
            proc.stdin.write(rgb.tobytes())
            frames += 1
    except BrokenPipeError:
        print("[record_ego] ffmpeg pipe broke unexpectedly", file=sys.stderr)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait(timeout=30)
        elapsed = time.monotonic() - t_start
        if rc == 0:
            print(f"[record_ego] Stopped. {frames} frames in {elapsed:.1f}s → {out_path}")
        else:
            print(f"[record_ego] ffmpeg exited rc={rc}. File may be truncated: {out_path}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
