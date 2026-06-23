"""【在机器人 Jetson 上运行】直接抓一帧 RealSense D435i 图像并保存, 用于排查。

绕过网络和 JPEG, 直接从相机取原始帧保存为 PNG, 便于判断问题出在
相机本身还是传输环节。会做几帧预热(让自动曝光稳定)再保存。

依赖: cam 环境 (pyrealsense2 + numpy + py-opencv)
运行:
    conda activate cam
    python capture_frame.py                 # 默认 640x480@30, 彩色+深度
    python capture_frame.py --width 1280 --height 720 --fps 30
    python capture_frame.py --no-depth
输出文件(保存到脚本所在目录的 captures/ 下):
    color.png       原始彩色图(BGR, cv2 可正常查看)
    depth.png       16bit 原始深度(单位 mm)
    depth_vis.png   深度伪彩色, 方便肉眼看
"""

import os
import argparse

import numpy as np
import cv2
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--warmup", type=int, default=30, help="预热帧数(等自动曝光稳定)")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    os.makedirs(out_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    # 打印设备 / 流信息
    dev = profile.get_device()
    print("设备:", dev.get_info(rs.camera_info.name),
          "| 序列号:", dev.get_info(rs.camera_info.serial_number),
          "| 固件:", dev.get_info(rs.camera_info.firmware_version))

    color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_prof.get_intrinsics()
    print(f"彩色: {intr.width}x{intr.height}  fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}  畸变={intr.model}")

    align = rs.align(rs.stream.color) if not args.no_depth else None
    if not args.no_depth:
        depth_scale = dev.first_depth_sensor().get_depth_scale()
        print(f"深度单位: 1 = {depth_scale*1000:.3f} mm (z16 原始值乘以它得到米)")

    try:
        print(f"预热 {args.warmup} 帧...")
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        if align is not None:
            frames = align.process(frames)

        color = frames.get_color_frame()
        color_img = np.asanyarray(color.get_data())
        color_path = os.path.join(out_dir, "color.png")
        cv2.imwrite(color_path, color_img)
        print(f"已保存彩色: {color_path}  shape={color_img.shape} dtype={color_img.dtype}")
        print(f"  彩色像素范围: min={color_img.min()} max={color_img.max()} mean={color_img.mean():.1f}")

        if not args.no_depth:
            depth = frames.get_depth_frame()
            depth_img = np.asanyarray(depth.get_data())
            depth_path = os.path.join(out_dir, "depth.png")
            cv2.imwrite(depth_path, depth_img)
            valid = depth_img[depth_img > 0]
            vmin = int(valid.min()) if valid.size else 0
            vmax = int(valid.max()) if valid.size else 0
            print(f"已保存深度: {depth_path}  shape={depth_img.shape} dtype={depth_img.dtype}")
            print(f"  有效深度范围: {vmin} ~ {vmax} mm  (有效像素 {valid.size}/{depth_img.size})")

            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            vis_path = os.path.join(out_dir, "depth_vis.png")
            cv2.imwrite(vis_path, depth_vis)
            print(f"已保存深度伪彩色: {vis_path}")

        print("\n完成。把 captures/ 里的图拷回上位机查看 (scp), 或用 scp 下载。")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
