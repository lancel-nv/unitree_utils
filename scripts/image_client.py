"""【在上位机运行】从机器人接收 RealSense D435i 图像并显示。

配合机器人端的 image_server.py 使用。彩色用 JPEG 解码显示，
深度(16bit, 单位 mm)解码后用伪彩色显示，并在鼠标位置显示距离。

上位机依赖(已在 g1 环境装好): pyzmq, opencv-python, numpy
运行：
    conda activate g1
    python image_client.py                        # 默认连 192.168.123.164:5555
    python image_client.py --host 192.168.123.164 --port 5555
按 q 退出。
"""

import argparse

import numpy as np
import cv2
import zmq

_last_depth = None


def _on_mouse(event, x, y, flags, param):
    if _last_depth is not None and 0 <= y < _last_depth.shape[0] and 0 <= x < _last_depth.shape[1]:
        if event == cv2.EVENT_MOUSEMOVE:
            d = _last_depth[y, x]
            param[0] = f"({x},{y}) = {d} mm" if d > 0 else f"({x},{y}) = 无效"


def main():
    global _last_depth
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164", help="机器人 Jetson 的 IP")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    # 注意: 不能用 zmq.CONFLATE, 它不支持 multipart 消息会导致收不到帧。
    # 改为小 HWM + 每次循环排空积压帧, 同样能做到低延迟只显示最新帧。
    sock.setsockopt(zmq.RCVHWM, 5)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    addr = f"tcp://{args.host}:{args.port}"
    sock.connect(addr)
    print(f"[client] 连接 {addr}, 等待图像... (窗口里按 q 退出)")
    got_first = False

    info = [""]
    cv2.namedWindow("color")
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", _on_mouse, info)

    while True:
        # 阻塞收一帧, 再把积压的帧全部排空, 只保留最新一帧 -> 低延迟
        parts = sock.recv_multipart()
        while True:
            try:
                parts = sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break

        if len(parts) != 4:
            continue
        if not got_first:
            print("[client] 已收到图像流。")
            got_first = True
        _, ts, color_buf, depth_buf = parts

        color = cv2.imdecode(np.frombuffer(color_buf, np.uint8), cv2.IMREAD_COLOR)
        if color is not None:
            cv2.imshow("color", color)
            # cv2.imwrite("color_720p.png", color)

        if depth_buf:
            depth = cv2.imdecode(np.frombuffer(depth_buf, np.uint8), cv2.IMREAD_UNCHANGED)
            _last_depth = depth
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            if info[0]:
                cv2.putText(depth_vis, info[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
            cv2.imshow("depth", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
