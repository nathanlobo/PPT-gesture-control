import argparse
import socket
import time
import signal

import cv2
import imagezmq
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageZMQ camera sender.")
    parser.add_argument(
        "--connect",
        default="tcp://127.0.0.1:5555",
        help="ImageHub endpoint, e.g. tcp://192.168.1.50:5555",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or video file path.",
    )
    parser.add_argument(
        "--name",
        default=socket.gethostname(),
        help="Sender name shown on the receiver side.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Resize width (0 = keep original).",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show local preview window.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Limit send rate (0 = unlimited).",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        return cap
    return cv2.VideoCapture(source)


def create_sender(connect_to: str) -> imagezmq.ImageSender:
    sender = imagezmq.ImageSender(connect_to=connect_to)
    sender.zmq_socket.setsockopt(zmq.LINGER, 0)
    # Avoid hanging forever if receiver is down (REQ/REP waits for reply).
    sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 500)
    sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 500)
    # Allow recovery if a recv times out (REQ sockets normally can't send again until recv happens).
    try:
        sender.zmq_socket.setsockopt(zmq.REQ_RELAXED, 1)
        sender.zmq_socket.setsockopt(zmq.REQ_CORRELATE, 1)
    except Exception:
        pass
    return sender


def main() -> int:
    args = parse_args()
    sender = create_sender(args.connect)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    window_name = f"Sender ({args.name} -> {args.connect})"
    last_send = 0.0
    fps_interval = (1.0 / args.fps) if args.fps and args.fps > 0 else 0.0

    frames = 0
    t0 = time.perf_counter()
    last_fps = 0.0
    stop = False

    def _handle_sigint(_sig: int, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    if args.display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV GUI functions are not available (cv2.namedWindow failed). "
                "If you installed opencv-python-headless, uninstall it and install opencv-python."
            ) from exc

    try:
        while not stop:
            ok, frame = cap.read()
            if not ok:
                break

            if args.width and args.width > 0:
                h, w = frame.shape[:2]
                new_w = args.width
                new_h = int(h * (new_w / w))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if fps_interval:
                now = time.perf_counter()
                sleep_s = (last_send + fps_interval) - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                last_send = time.perf_counter()

            frames += 1
            if frames % 30 == 0:
                dt = time.perf_counter() - t0
                if dt > 0:
                    last_fps = frames / dt
                    print(f"sent_fps={last_fps:.1f}")

            if args.display:
                preview = frame
                cv2.putText(
                    preview,
                    f"{args.name}  ->  {args.connect}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    f"sent_fps={last_fps:.1f}   |   press 'q' to quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                try:
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV GUI functions are not available (cv2.imshow failed). "
                        "If you installed opencv-python-headless, uninstall it and install opencv-python."
                    ) from exc

            # Send after preview so you still get local feedback even if the receiver is down
            # (REQ/REP mode blocks until the receiver replies).
            try:
                sender.send_image(args.name, frame)
            except (zmq.error.Again, zmq.error.ZMQError):
                # Receiver didn't reply within timeout (or isn't reachable), or REQ socket is in a bad FSM state.
                # Recreate the sender socket so we can recover when the receiver returns.
                try:
                    sender.close()
                except Exception:
                    pass
                sender = create_sender(args.connect)
                continue
    except KeyboardInterrupt:
        stop = True
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        try:
            sender.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
