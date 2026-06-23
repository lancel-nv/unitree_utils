"""【在机器人 Jetson 上运行】读取 RealSense D435i，通过 ZMQ 把图像推送到上位机。

D435i 通过 USB 接在 Jetson(192.168.123.164) 上，本脚本在 Jetson 上跑，
把彩色(JPEG) + 深度(16bit PNG) 打包用 ZMQ PUB 发出，上位机用 image_client.py 接收。

机器人端依赖安装(推荐, 因为 Jetson 上 pip 装不了 pyrealsense2)：
    conda create -y -n cam -c conda-forge python=3.10 librealsense numpy py-opencv pyzmq
    conda activate cam
运行：
    python image_server.py                 # 默认 640x480@30, 彩色+深度, 端口 5555
    python image_server.py --no-depth       # 只发彩色(带宽更低)
    python image_server.py --width 1280 --height 720 --fps 15
"""

import argparse
import time

import numpy as np
import cv2
import zmq
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-depth", action="store_true", help="只发彩色, 不发深度")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    args = ap.parse_args()

    # ---- 启动 RealSense ----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline.start(config)
    # 把深度对齐到彩色坐标系, 方便上位机做 RGB-D
    align = rs.align(rs.stream.color) if not args.no_depth else None

    # ---- ZMQ 发布 ----
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)        # 只保留最新帧, 降延迟
    sock.bind(f"tcp://*:{args.port}")
    print(f"[server] D435i 推流 tcp://*:{args.port}  分辨率={args.width}x{args.height}@{args.fps} "
          f"深度={'关' if args.no_depth else '开'}  (Ctrl-C 退出)")

    jpg_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color = frames.get_color_frame()
            if not color:
                continue
            color_img = np.asanyarray(color.get_data())
            ok, color_jpg = cv2.imencode(".jpg", color_img, jpg_param)
            if not ok:
                continue
            ts = np.array([time.time()], dtype=np.float64).tobytes()

            if not args.no_depth:
                depth = frames.get_depth_frame()
                depth_img = np.asanyarray(depth.get_data())   # uint16, 单位 mm
                ok2, depth_png = cv2.imencode(".png", depth_img)
                depth_buf = depth_png.tobytes() if ok2 else b""
            else:
                depth_buf = b""

            sock.send_multipart([b"frame", ts, color_jpg.tobytes(), depth_buf])
    except KeyboardInterrupt:
        print("\n[server] 退出")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
