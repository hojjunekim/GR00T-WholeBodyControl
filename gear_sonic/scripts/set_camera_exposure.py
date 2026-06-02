"""CLI to toggle camera options at runtime on a running ComposedCameraSensor.

The camera server (launched as `composed_camera_server.service` on the Jetson)
binds a ZMQ REP socket on `port + 1` (default 5556) for runtime commands. This
script sends a msgpack request and prints the reply.

Examples
--------
Turn auto-exposure ON:
    python -m gear_sonic.scripts.set_camera_exposure --host 192.168.123.164 --auto on

Turn auto-exposure OFF with a manual exposure of 8500us:
    python -m gear_sonic.scripts.set_camera_exposure --host 192.168.123.164 \\
        --auto off --exposure-us 8500

List running cameras:
    python -m gear_sonic.scripts.set_camera_exposure --host 192.168.123.164 --list
"""

import argparse
import sys

import msgpack
import zmq


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="192.168.123.164",
                   help="Camera server host (Jetson IP)")
    p.add_argument("--port", type=int, default=5556,
                   help="Control REP port (defaults to camera port + 1 = 5556)")
    p.add_argument("--camera", default="ego_view",
                   help="Mount position to target (default: ego_view)")
    p.add_argument("--auto", choices=["on", "off"],
                   help="Toggle auto-exposure on/off")
    p.add_argument("--exposure-us", type=int,
                   help="Manual exposure in microseconds (used with --auto off)")
    p.add_argument("--awb", choices=["on", "off"],
                   help="Toggle auto-white-balance on/off")
    p.add_argument("--white-balance-k", type=int,
                   help="Manual white balance in Kelvin (used with --awb off)")
    p.add_argument("--get", action="store_true",
                   help="Print the live AE/exposure/AWB/gain values from the sensor")
    p.add_argument("--get-intrinsics", action="store_true",
                   help="Print color/depth intrinsics (fx, fy, cx, cy, distortion) "
                        "and depth->color extrinsics from the live pipeline")
    p.add_argument("--list", action="store_true",
                   help="List the cameras the server knows about")
    p.add_argument("--timeout-ms", type=int, default=2000,
                   help="REQ recv timeout (ms)")
    args = p.parse_args()

    if not (args.list or args.get or args.get_intrinsics or args.auto or args.awb):
        p.error("must pass --auto, --awb, --get, --get-intrinsics, or --list")

    if args.list:
        request = {"cmd": "list_cameras"}
    elif args.get_intrinsics:
        request = {"cmd": "get_intrinsics", "camera": args.camera}
    elif args.get:
        request = {"cmd": "get_status", "camera": args.camera}
    elif args.awb is not None and args.auto is None:
        request = {"cmd": "set_white_balance", "camera": args.camera,
                   "auto": args.awb == "on"}
        if args.white_balance_k is not None:
            request["kelvin"] = int(args.white_balance_k)
    else:
        request = {
            "cmd": "set_exposure",
            "camera": args.camera,
            "auto": args.auto == "on",
        }
        if args.exposure_us is not None:
            request["exposure_us"] = int(args.exposure_us)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.send(msgpack.packb(request, use_bin_type=True))
    try:
        raw = sock.recv()
    except zmq.Again:
        print(f"ERROR: no reply within {args.timeout_ms}ms — is the camera server running?",
              file=sys.stderr)
        sys.exit(1)

    reply = msgpack.unpackb(raw, raw=False)
    if args.get_intrinsics and reply.get("ok") and "intrinsics" in reply:
        import json
        print(json.dumps(reply["intrinsics"], indent=2))
    else:
        print(reply)
    if not reply.get("ok", False):
        sys.exit(2)


if __name__ == "__main__":
    main()
