"""
【上位机运行】拍照 + 计算 G1 pelvis 相对于 ChArUco 标定板的位姿。

整体思路 (链式坐标变换)
------------------------
目标: 求 pelvis 在标定板坐标系下的位姿  T_board_pelvis。

1) 拍照: 复用 `1.g1_left_arm_teach_capture_remote.py` 里的 ZmqCamera, 从 Jetson 的
   image_server.py (ZMQ 推流) 收一帧彩色图, 保存到本地。
2) 相机 ←→ 板子: 图里有一块 ChArUco 板, 用其上多个二维码做 solvePnP, 得到板子在
   【相机光心系】下的位姿  T_camopt_board。
3) 相机 ←→ pelvis: 相机固定在 torso 上, 从 URDF 正运动学 (pinocchio) 求
   d435_link 在 pelvis 下的位姿, 再叠加 "相机body系→光心系" 的固定旋转, 得到
   T_pelvis_camopt。腰部 3 个关节会影响该位姿: 若给了 --net 则从 rt/lowstate 读
   实际腰关节角, 否则按 0 处理。
4) 串起来:
       T_pelvis_board = T_pelvis_camopt @ T_camopt_board
       T_board_pelvis = inv(T_pelvis_board)        # 这就是要的结果

前置条件
--------
- Jetson 端先推流:   conda activate cam && python scripts/image_server.py   (默认 640x480)
- 相机内参 json (须与推流分辨率一致), 默认用 workflow/intrinsics_640x480.json,
  没有就用 scripts/dump_intrinsics.py 在 Jetson 导出后 scp 回来。
- ChArUco 板的参数 (格子数 / 方格边长 / 二维码边长 / 字典) 必须和实际打印的板一致!

用法
----
    # 默认: 弹预览窗口, 看好按 c 拍照算位姿, q 退出; 腰关节按 0
    python3 4.capture_image.py \
        --squares-x 5 --squares-y 7 --square-len 0.035 --marker-len 0.026 --dict DICT_5X5_100

    # 想让腰关节用真实角度 (更准): 加 --net 连机器人读 rt/lowstate
    python3 4.capture_image.py --net enp12s0 ...

    # 无显示器/SSH 无 X: 加 --no-preview, 自动抓一帧直接算
    python3 4.capture_image.py --no-preview ...
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import cv2
import pinocchio as pin


# ---------------------------------------------------------------------------
# 复用既有代码 (文件名以数字开头/含点, 不能直接 import, 用 importlib 按路径加载)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ZmqCamera: 与采集脚本同一套收帧/排空逻辑 (低延迟只取最新帧)
_teach = _load_module(
    "teach_capture_remote",
    os.path.join(_HERE, "1.g1_left_arm_teach_capture_remote.py"),
)
ZmqCamera = _teach.ZmqCamera


# ---------------------------------------------------------------------------
# 固定标定常量
# ---------------------------------------------------------------------------
# URDF 里 d435_link 是相机 body 系 (x 朝前, y 朝左, z 朝上)。
# OpenCV/solvePnP 给出的是相机【光心系】(x 朝右, y 朝下, z 朝前/射向场景)。
# 二者差一个固定旋转 (等价 ROS realsense 描述里 rpy=(-pi/2, 0, -pi/2)):
#   x_opt = -y_link, y_opt = -z_link, z_opt = x_link
_R_LINK_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])
T_D435LINK_OPTICAL = np.eye(4)
T_D435LINK_OPTICAL[:3, :3] = _R_LINK_OPTICAL

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_SDK_IDX = [12, 13, 14]   # rt/lowstate.motor_state 下标 (yaw/roll/pitch)

_DEFAULT_URDF = os.path.join(_ROOT, "g1_description", "g1_29dof_with_hand_rev_1_0.urdf")
_DEFAULT_INTR = os.path.join(_HERE, "intrinsics_640x480.json")


# ---------------------------------------------------------------------------
# 内参 / ChArUco
# ---------------------------------------------------------------------------
def load_intrinsics(path):
    with open(path) as f:
        intr = json.load(f)
    K = np.array([[intr["fx"], 0.0, intr["ppx"]],
                  [0.0, intr["fy"], intr["ppy"]],
                  [0.0, 0.0, 1.0]])
    dist = np.array(intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    return K, dist, intr


def build_charuco(args):
    if not hasattr(cv2.aruco, args.dict):
        raise SystemExit(f"未知的 aruco 字典: {args.dict}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_len, args.marker_len, aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def detect_board_pose(img, board, detector, K, dist):
    """检测 ChArUco 板, 返回 (T_camopt_board, 用于可视化的角点信息) 或 None。

    T_camopt_board: 标定板坐标系在【相机光心系】下的 4x4 位姿。
    板子坐标系原点在棋盘左下角, z 垂直板面朝外, 与 cv2.aruco.CharucoBoard 一致。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 4:
        return None
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return {
        "T": T, "rvec": rvec, "tvec": tvec,
        "ch_corners": ch_corners, "ch_ids": ch_ids,
        "m_corners": m_corners, "m_ids": m_ids,
        "n_corners": len(ch_ids),
    }


def draw_detection(img, det, K, dist, axis_len):
    vis = img.copy()
    if det["m_ids"] is not None:
        cv2.aruco.drawDetectedMarkers(vis, det["m_corners"], det["m_ids"])
    if det["ch_ids"] is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, det["ch_corners"], det["ch_ids"])
    cv2.drawFrameAxes(vis, K, dist, det["rvec"], det["tvec"], axis_len)
    return vis


# ---------------------------------------------------------------------------
# URDF 正运动学: pelvis -> 相机光心
# ---------------------------------------------------------------------------
def build_kinematics(urdf_path):
    model = pin.buildModelFromUrdf(urdf_path)   # 根 (固定基) = pelvis
    data = model.createData()
    if not model.existFrame("d435_link") or not model.existFrame("pelvis"):
        raise SystemExit("URDF 里找不到 d435_link / pelvis frame")
    return model, data


def pelvis_to_camopt(model, data, waist_q):
    """waist_q: {关节名: 角度rad}; 返回 T_pelvis_camopt (相机光心在 pelvis 下)。"""
    q = pin.neutral(model)
    for name, val in waist_q.items():
        if name in model.names:
            q[model.joints[model.getJointId(name)].idx_q] = float(val)
    pin.framesForwardKinematics(model, data, q)
    T_pelvis_d435 = data.oMf[model.getFrameId("d435_link")].homogeneous.copy()
    return T_pelvis_d435 @ T_D435LINK_OPTICAL


def read_waist_from_lowstate(net, timeout_s=3.0):
    """连 DDS 读一帧 rt/lowstate, 取腰部 3 关节角; 失败返回全 0。只读, 安全。"""
    try:
        from unitree_sdk2py.core.channel import (
            ChannelSubscriber, ChannelFactoryInitialize,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ImportError:
        print("WARNING: 未安装 unitree_sdk2py, 腰关节按 0 处理。")
        return {n: 0.0 for n in WAIST_JOINTS}

    ChannelFactoryInitialize(0, net)
    holder = {"s": None}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: holder.__setitem__("s", m), 10)
    deadline = time.time() + timeout_s
    while holder["s"] is None and time.time() < deadline:
        time.sleep(0.05)
    if holder["s"] is None:
        print("WARNING: 超时未收到 rt/lowstate, 腰关节按 0 处理。")
        return {n: 0.0 for n in WAIST_JOINTS}
    s = holder["s"]
    waist = {n: float(s.motor_state[idx].q) for n, idx in zip(WAIST_JOINTS, WAIST_SDK_IDX)}
    print("腰关节角(rad):", {k: round(v, 4) for k, v in waist.items()})
    return waist


# ---------------------------------------------------------------------------
# 位姿格式化 / 输出
# ---------------------------------------------------------------------------
def pose_summary(T):
    """4x4 位姿 -> dict: 平移(m) / rpy(deg) / 四元数(wxyz) / 原始矩阵。"""
    R = T[:3, :3]
    t = T[:3, 3]
    rpy = pin.rpy.matrixToRpy(np.asarray(R)) * 180.0 / np.pi
    quat = pin.Quaternion(np.asarray(R))
    return {
        "translation_m": [float(v) for v in t],
        "rpy_deg": [float(v) for v in rpy],
        "quat_wxyz": [float(quat.w), float(quat.x), float(quat.y), float(quat.z)],
        "matrix": [[float(v) for v in row] for row in T],
    }


def print_pose(name, T):
    s = pose_summary(T)
    print(f"\n=== {name} ===")
    print("  平移 xyz (m):  " + ", ".join(f"{v:+.4f}" for v in s["translation_m"]))
    print("  姿态 rpy(deg): " + ", ".join(f"{v:+.2f}" for v in s["rpy_deg"]))
    print("  四元数 wxyz:   " + ", ".join(f"{v:+.4f}" for v in s["quat_wxyz"]))
    with np.printoptions(precision=4, suppress=True):
        print("  矩阵:\n" + str(np.asarray(T)))


# ---------------------------------------------------------------------------
# 取一帧 (预览或直接抓)
# ---------------------------------------------------------------------------
def grab_frame(cam, no_preview):
    if no_preview:
        img = cam.capture()
        if img is None:
            raise SystemExit("未取到图像帧 (相机流异常?)")
        return img

    win = "capture (c=拍照算位姿  q=退出)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last = None
    try:
        while True:
            frame = cam.capture(timeout_ms=150)
            if frame is not None:
                last = frame
            if last is not None:
                disp = last.copy()
                cv2.putText(disp, "c: capture   q: quit", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
                cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and last is not None:
                return last
            if key in (ord("q"), ord("x"), 27):
                return None
    finally:
        cv2.destroyWindow(win)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # 相机 / 内参
    ap.add_argument("--host", default="192.168.123.164", help="Jetson(image_server) IP")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--intrinsics", default=_DEFAULT_INTR, help="相机内参 json (须与推流分辨率一致)")
    ap.add_argument("--cam-timeout", type=float, default=8.0, help="等待图像流超时(s)")
    ap.add_argument("--no-preview", action="store_true", help="不开预览窗口, 直接抓一帧")
    # ChArUco 板参数 (按实际打印的板填!)
    ap.add_argument("--squares-x", type=int, default=5, help="ChArUco 横向方格数")
    ap.add_argument("--squares-y", type=int, default=7, help="ChArUco 纵向方格数")
    ap.add_argument("--square-len", type=float, default=0.035, help="方格边长 (m)")
    ap.add_argument("--marker-len", type=float, default=0.026, help="二维码边长 (m)")
    ap.add_argument("--dict", default="DICT_5X5_100", help="aruco 字典名 (如 DICT_5X5_100)")
    # URDF / 机器人
    ap.add_argument("--urdf", default=_DEFAULT_URDF, help="G1 URDF (含 d435_link)")
    ap.add_argument("--net", default=None,
                    help="网卡名; 给了才连 rt/lowstate 读真实腰关节角, 否则腰按 0")
    # 输出
    ap.add_argument("--out", default=None, help="输出目录, 默认 ./pose_to_board_<时间戳>")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(_HERE, f"pose_to_board_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.out, exist_ok=True)

    K, dist, intr = load_intrinsics(args.intrinsics)
    board, detector = build_charuco(args)
    model, data = build_kinematics(args.urdf)

    # 1) 拍照
    cam = ZmqCamera(args.host, args.port, args.intrinsics)
    print(f"检测相机流 (最多 {args.cam_timeout:.0f}s) ...", flush=True)
    if not cam.wait_for_stream(args.cam_timeout):
        cam.stop()
        raise SystemExit("ERROR: 未收到图像流, 请确认 Jetson image_server.py 已启动。")
    print("相机流正常。")
    img = grab_frame(cam, args.no_preview)
    cam.stop()
    if img is None:
        print("已退出, 未拍照。")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(args.out, f"{stamp}_raw.png")
    cv2.imwrite(raw_path, img)
    print(f"已保存原图: {raw_path}")

    # 2) 相机 ←→ 板子
    det = detect_board_pose(img, board, detector, K, dist)
    if det is None:
        print("ERROR: 未检测到 ChArUco 板 (检查板参数 / 光照 / 是否入镜)。")
        return
    print(f"检测到 ChArUco 角点 {det['n_corners']} 个。")
    T_camopt_board = det["T"]

    vis = draw_detection(img, det, K, dist, args.square_len * 2)
    vis_path = os.path.join(args.out, f"{stamp}_detected.png")
    cv2.imwrite(vis_path, vis)
    print(f"已保存检测可视化: {vis_path}")

    # 3) 相机 ←→ pelvis (URDF FK; 腰关节可选实读)
    waist_q = read_waist_from_lowstate(args.net) if args.net else \
        {n: 0.0 for n in WAIST_JOINTS}
    if not args.net:
        print("提示: 未给 --net, 腰关节按 0 处理 (锁定站立且腰回正时 OK)。")
    T_pelvis_camopt = pelvis_to_camopt(model, data, waist_q)

    # 4) 串联
    T_pelvis_board = T_pelvis_camopt @ T_camopt_board
    T_board_pelvis = np.linalg.inv(T_pelvis_board)

    print_pose("T_camopt_board  (板子 在 相机光心系)", T_camopt_board)
    print_pose("T_pelvis_camopt (相机光心 在 pelvis)", T_pelvis_camopt)
    print_pose("T_pelvis_board  (板子 在 pelvis)", T_pelvis_board)
    print_pose(">>> T_board_pelvis (pelvis 在 板子坐标系) <<<", T_board_pelvis)

    result = {
        "timestamp": stamp,
        "image_raw": os.path.basename(raw_path),
        "image_detected": os.path.basename(vis_path),
        "intrinsics": intr,
        "charuco": {
            "squares_x": args.squares_x, "squares_y": args.squares_y,
            "square_len": args.square_len, "marker_len": args.marker_len,
            "dict": args.dict, "n_corners": int(det["n_corners"]),
        },
        "waist_q_rad": waist_q,
        "T_camopt_board": pose_summary(T_camopt_board),
        "T_pelvis_camopt": pose_summary(T_pelvis_camopt),
        "T_pelvis_board": pose_summary(T_pelvis_board),
        "T_board_pelvis": pose_summary(T_board_pelvis),
    }
    res_path = os.path.join(args.out, f"{stamp}_pose.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已保存结果: {res_path}")


if __name__ == "__main__":
    main()
