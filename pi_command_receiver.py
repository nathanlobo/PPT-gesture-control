import argparse
import json
import signal
import time

import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive gesture commands over ZMQ and trigger hardware actions."
    )
    parser.add_argument(
        "--connect",
        default="tcp://127.0.0.1:6000",
        help="ZMQ PUB endpoint from the gesture machine, e.g. tcp://192.168.1.50:6000",
    )
    parser.add_argument(
        "--topic",
        default="gesture",
        help="ZMQ topic to subscribe to.",
    )
    return parser.parse_args()


def handle_command(cmd: str, payload: dict) -> None:
    # TODO: Replace this with your hardware control code (GPIO / serial / motors).
    # Keep it fast; if you need longer work, queue it and return quickly.
    print(f"{time.strftime('%H:%M:%S')} cmd={cmd} payload={payload}")


def main() -> int:
    args = parse_args()
    stop = False

    def _handle_sigint(_sig: int, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.connect(args.connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    print(f"Listening: {args.connect} topic={args.topic} (Ctrl+C to stop)")

    try:
        while not stop:
            try:
                topic, msg = sock.recv_multipart()
            except zmq.error.Again:
                continue

            try:
                payload = json.loads(msg.decode("utf-8"))
            except Exception:
                payload = {"raw": msg.decode("utf-8", errors="replace")}

            cmd = str(payload.get("cmd", "")).strip().upper()
            if not cmd:
                continue

            handle_command(cmd, payload)
    except KeyboardInterrupt:
        stop = True
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

