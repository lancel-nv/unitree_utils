"""
G1 左臂【拖动示教 + 拍照采集】—— 上位机版（相机走 ZMQ，本机不插相机；独立脚本）。

与方法一 `g1_left_arm_teach_capture.py` 思路一致，但：
- 不依赖该文件，逻辑全部内联（本机无需 pyrealsense2）。
- 相机改为从机器人 Jetson 上的 `scripts/image_server.py` ZMQ 推流接收彩色帧
  （收帧/排空思路同 `scripts/image_client.py`）。

核心控制逻辑
------------
- 通过 `rt/arm_sdk` 叠加接管上半身；腿不发指令，继续由锁定站立控制器管。
- 腰(12-14) + 右臂(22-28)：大刚度死锁在开机角度。
- 左臂(15-21)：阻尼软模式（kp→0、阻尼 kd、无重力补偿），可用手拖动；松手会缓慢
  下垂，需扶住。
- 三段式安全启动（HOLD→BLEND→SOFT），防止切软瞬间掉臂。
- 按 c 采样：在输出目录存一对同名时间戳文件 —— <时间戳>.png + <时间戳>.json,
  json 内含左臂关节状态 (joint_states) 和末端位姿 (eef_pose, FK 自实测角)。
  另外启动时写一次 intrinsics.json (相机内参)。

适用场景
--------
- 采集脚本跑在【上位机】(有 unitree_sdk2py + pinocchio，DDS 直连机器人主控)。
- D435i 插在【Jetson 192.168.123.164】上，Jetson 跑 `image_server.py` 推流。
和
前置条件
--------
1. 机器人锁定站立 (锁定站立 = L2 + Up)，不要释放运动模式。
2. 标定板牢固固定在【左手】上 (eye-to-hand)，启动时先用手扶住左臂。
3. 上位机 g1 环境需要 pinocchio：  pip install pin   (zmq/opencv/numpy 已具备)
4. Jetson 端先启动推流 (注意分辨率, 决定内参)：
       conda activate cam && python image_server.py            # 默认 640x480@30
5. 相机内参 intrinsics.json：在 Jetson 用同目录 dump_intrinsics.py 按"推流分辨率"
   导出后 scp 回上位机，用 --intrinsics 传入 (solve_handeye 必需)。

用法
----
python3 1.g1_left_arm_teach_capture_remote.py  --ee-frame left_hand_palm_link --intrinsics ./intrinsics_640x480.json

启动流程（安全）
----------------
1. 脚本先连 DDS、建运动学，再探测相机流（--cam-timeout，默认 8s）。
2. 相机流没通 -> 打印 ERROR 并安全退出，【不接管手臂】。
3. 相机流正常 -> 按 Enter 手动确认，才开始接管左臂进入软模式。

操作（默认开实时预览窗口）
--------------------------
- 弹出 OpenCV 窗口实时显示相机画面, 叠加当前模式 / 末端位置 / 已存数量。
- 看满意了在窗口里按 c 拍照(存当前预览帧 + 关节角 + FK 末端位姿), q 或 x 退出。
- 加 --no-preview 可回退到纯终端键盘模式(无显示器/SSH 无 X 时): 终端里 c 拍照, x 退出。
"""

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import cv2
import zmq
import pinocchio as pin

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# --- G1 joint layout (29 DoF): URDF joint name -> SDK motor index -------------
LEFT_ARM_JOINTS = {
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
}
# joints held stiff: waist + right arm (legs are NOT commanded -> stay with the
# onboard standing controller; head has no joints)
HELD_SDK_IDX = [12, 13, 14] + list(range(22, 29))
WEIGHT_IDX = 29   # arm_sdk enable flag (motor_cmd[29].q)

# every controllable joint name -> SDK index, for building the full-body FK config
ALL_SDK = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    **LEFT_ARM_JOINTS,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

CONTROL_DT = 0.01    # 100 Hz
KP_HOLD = 80.0       # stiffness for locked joints + initial left-arm hold
KD_HOLD = 2.0
KD_SOFT = 2.0        # damping for left arm in free (teach) mode (no gravity comp)
T_ENGAGE = 1.5       # s: ramp arm_sdk weight to 1 while holding pose
T_RELEASE = 2.0      # s: blend kp->0 into damping soft mode (no sudden drop)


class ZmqCamera:
    """从 Jetson 的 image_server.py (ZMQ PUB) 订阅彩色帧。

    image_server 发的是 4-part multipart: [topic, ts, color_jpg, depth_png]。
    内参不在流里，需通过 intrinsics_path 提供。仅在主线程使用 (capture 由
    keyboard_loop 调用)，符合 zmq socket 单线程约束。
    接口与本地相机一致：intrinsics() / capture() / stop()。
    """

    def __init__(self, host, port, intrinsics_path=None, timeout_ms=5000):
        self._intr = None
        if intrinsics_path:
            if os.path.isfile(intrinsics_path):
                with open(intrinsics_path) as f:
                    self._intr = json.load(f)
            else:
                print(f"WARNING: 内参文件不存在: {intrinsics_path}；将不写 intrinsics.json。"
                      "可先跑通流程，之后用 dump_intrinsics.py 在 Jetson 导出再补上。")

        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 5)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(f"tcp://{host}:{port}")
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        print(f"[zmq-cam] 连接 tcp://{host}:{port}", flush=True)

    def wait_for_stream(self, timeout_s=8.0):
        """阻塞探测图像流：收到一帧返回 True，超时返回 False（不抛异常）。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            remaining_ms = max(1, int((deadline - time.time()) * 1000))
            if self.poller.poll(min(remaining_ms, 500)):
                if self._recv_latest() is not None:
                    return True
        return False

    def _recv_latest(self, timeout_ms=None):
        """收一帧, 再排空积压, 只保留最新一帧 -> 低延迟。超时返回 None。"""
        if timeout_ms is None:
            timeout_ms = self.timeout_ms
        if not self.poller.poll(timeout_ms):
            return None
        parts = self.sock.recv_multipart()
        while True:
            try:
                parts = self.sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
        if len(parts) != 4:
            return None
        _, _, color_buf, _ = parts
        return cv2.imdecode(np.frombuffer(color_buf, np.uint8), cv2.IMREAD_COLOR)

    def intrinsics(self):
        return self._intr

    def capture(self, timeout_ms=None):
        return self._recv_latest(timeout_ms)

    def stop(self):
        self.sock.close(0)


class TeachCaptureRemote:
    def __init__(self, args):
        self.args = args
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.weight = 0.0
        self.t = 0.0
        self.running = True
        self.releasing = False           # 退出时置位: 平滑交还上半身, 防抖动
        self.left_release_hold = {}       # 交还时左臂定格的实测姿态
        self.records = []
        os.makedirs(args.out, exist_ok=True)

    # ---- DDS -----------------------------------------------------------------
    def init_dds(self):
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        print("Waiting for first rt/lowstate ...")
        while not self.first_state:
            time.sleep(0.1)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()

    def _on_state(self, msg: LowState_):
        self.low_state = msg
        self.first_state = True

    # ---- kinematics ----------------------------------------------------------
    def init_kinematics(self):
        self.model = pin.buildModelFromUrdf(self.args.urdf)  # fixed base @ pelvis
        self.fk_data = self.model.createData()

        missing = [n for n in LEFT_ARM_JOINTS if n not in self.model.names]
        if missing or not self.model.existFrame(self.args.ee_frame):
            print("Joint/frame not found. Available names:")
            print("Joints:", list(self.model.names)[1:])
            print("Frames:", [f.name for f in self.model.frames])
            sys.exit(1)

        self.ee_id = self.model.getFrameId(self.args.ee_frame)
        # joint -> (idx_q, idx_v, sdk) for fast config building / torque extraction
        self.qmap = [(self.model.joints[self.model.getJointId(n)].idx_q, sdk)
                     for n, sdk in ALL_SDK.items() if n in self.model.names]
        self.left_iv = [(self.model.joints[self.model.getJointId(n)].idx_v, sdk)
                        for n, sdk in LEFT_ARM_JOINTS.items()]
        # hold targets for locked joints + initial left-arm pose
        self.held = {sdk: self.low_state.motor_state[sdk].q for sdk in HELD_SDK_IDX}
        self.left_hold = {sdk: self.low_state.motor_state[sdk].q
                          for sdk in LEFT_ARM_JOINTS.values()}

    def _full_q(self):
        q = pin.neutral(self.model)
        for idx_q, sdk in self.qmap:
            q[idx_q] = self.low_state.motor_state[sdk].q
        return q

    def _measured_left(self):
        return [self.low_state.motor_state[sdk].q for sdk in LEFT_ARM_JOINTS.values()]

    def _measured_ee(self, q):
        pin.framesForwardKinematics(self.model, self.fk_data, q)
        return self.fk_data.oMf[self.ee_id].copy()

    # ---- control loop (100 Hz) -----------------------------------------------
    def control_step(self):
        if self.low_state is None:
            return
        self.t += CONTROL_DT

        # weight: 接管时 0->1 渐入; 退出时 1->0 渐出(平滑交还站立控制器, 防抖动)
        if self.releasing:
            self.weight = max(0.0, self.weight - CONTROL_DT / T_ENGAGE)
        else:
            self.weight = min(1.0, self.weight + CONTROL_DT / T_ENGAGE)

        # blend factor: 0 = stiff hold, 1 = full soft (damping only, no gravity comp)
        if self.t <= T_ENGAGE:
            alpha = 0.0
        else:
            alpha = np.clip((self.t - T_ENGAGE) / T_RELEASE, 0.0, 1.0)

        self.low_cmd.motor_cmd[WEIGHT_IDX].q = self.weight

        # left arm (纯阻尼拖动, 无重力补偿: tau 恒为 0)
        for _iv, sdk in self.left_iv:
            m = self.low_cmd.motor_cmd[sdk]
            if self.releasing:
                # 交还阶段: 用退出瞬间的实测姿态硬保持, 配合 weight 渐出平滑交还
                m.q = float(self.left_release_hold[sdk])
                m.dq = 0.0
                m.kp = KP_HOLD
                m.kd = KD_HOLD
                m.tau = 0.0
            else:
                # 正常: 由硬保持(alpha=0)渐变到阻尼软模式(alpha=1)
                m.q = float(self.left_hold[sdk])      # only matters while kp>0
                m.dq = 0.0
                m.kp = KP_HOLD * (1.0 - alpha)
                m.kd = KD_HOLD * (1.0 - alpha) + KD_SOFT * alpha
                m.tau = 0.0

        # waist + right arm: locked stiff
        for sdk, qh in self.held.items():
            m = self.low_cmd.motor_cmd[sdk]
            m.q, m.dq, m.tau, m.kp, m.kd = float(qh), 0.0, 0.0, KP_HOLD, KD_HOLD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def release(self):
        """退出前调用: 把左臂从软模式切回硬保持当前姿态, 并将 arm_sdk weight
        平滑降到 0, 交还上半身给站立控制器, 避免瞬间抖动/弹起。
        依赖控制线程仍在运行(由它逐步执行 weight 渐出)。"""
        if self.low_state is None:
            return
        self.left_release_hold = {sdk: self.low_state.motor_state[sdk].q
                                  for sdk in LEFT_ARM_JOINTS.values()}
        self.releasing = True
        print("\n交还上半身给站立控制器 (weight -> 0) ...")
        time.sleep(T_ENGAGE + 0.5)   # 等 weight 渐出到 0

    # ---- capture / io --------------------------------------------------------
    def _capture(self, img=None):
        if img is None:
            img = self.rs.capture()
        if img is None:
            print("\n[capture] no frame, skipped")
            return
        qL = self._measured_left()
        T = self._measured_ee(self._full_q())
        quat = pin.Quaternion(T.rotation)  # normalized

        # 同名时间戳: 一张 png + 一个 json (毫秒级, 避免同秒冲突)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
        img_path = os.path.join(self.args.out, stamp + ".png")
        json_path = os.path.join(self.args.out, stamp + ".json")
        cv2.imwrite(img_path, img)

        rec = {
            "timestamp": stamp,
            "image": os.path.basename(img_path),
            "joint_states": {
                "left_arm_names": list(LEFT_ARM_JOINTS.keys()),
                "left_arm_q": qL,
            },
            "eef_pose": {
                "frame": self.args.ee_frame,
                "R_base_ee": T.rotation.flatten().tolist(),
                "t_base_ee": T.translation.tolist(),
                "quat_wxyz": [quat.w, quat.x, quat.y, quat.z],
            },
        }
        with open(json_path, "w") as f:
            json.dump(rec, f, indent=2)
        self.records.append(rec)

        print(f"\n[capture] {stamp}  eef_pos={np.round(T.translation, 4)}  saved={len(self.records)}")

    def _mode_str(self):
        if self.t <= T_ENGAGE:
            return "HOLD"
        if self.t < T_ENGAGE + T_RELEASE:
            return "BLEND"
        return "SOFT(teach)"

    def _status(self):
        q = self._full_q()
        t = self._measured_ee(q).translation
        sys.stdout.write(
            f"\r[{self._mode_str()}] eef=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] "
            f"saved={len(self.records)}   ")
        sys.stdout.flush()

    def _overlay(self, img):
        """在预览帧上叠加状态文字 (模式 / 末端位置 / 已存数量 / 操作提示)。"""
        mode = self._mode_str()
        t = self._measured_ee(self._full_q()).translation
        ready = mode.startswith("SOFT")
        color = (0, 220, 0) if ready else (0, 200, 255)
        lines = [
            f"[{mode}]  saved={len(self.records)}",
            f"eef xyz=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m",
            "c: capture    q/x: quit" if ready else "engaging... (hold the arm)",
        ]
        y = 26
        for s in lines:
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
            y += 30
        return img

    STALE_S = 0.7   # 超过这个时间没收到新帧, 视为掉线/过期, 禁止拍照

    def preview_loop(self):
        """实时预览相机画面, 看满意了按 c 拍照, q/x 退出。

        用短超时轮询取帧, 即使相机流中断窗口也保持响应(可随时按 x 退出);
        流中断时显示 NO SIGNAL, 并禁止用过期帧拍照。"""
        win = "G1 left-arm teach"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last = None
        last_ts = 0.0
        while self.running:
            frame = self.rs.capture(timeout_ms=150)   # 短超时, 掉线也不卡死
            if frame is not None:
                last, last_ts = frame, time.time()
            fresh = last is not None and (time.time() - last_ts) < self.STALE_S
            if last is not None:
                disp = self._overlay(last.copy())
                if not fresh:
                    cv2.putText(disp, "NO SIGNAL", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow(win, disp)
            else:
                blank = np.zeros((240, 320, 3), np.uint8)
                cv2.putText(blank, "NO SIGNAL", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow(win, blank)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if fresh:
                    self._capture(last)          # 拍下当前预览到的这一帧
                else:
                    print("\n[capture] 无实时图像(掉线/过期), 已跳过")
            elif key in (ord("x"), ord("q"), 27):
                self.running = False
                break
        cv2.destroyWindow(win)

    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                dr, _, _ = select.select([sys.stdin], [], [], 0.2)
                if dr:
                    key = sys.stdin.read(1)
                    if key == "c":
                        self._capture()
                    elif key in ("x", "\x1b", "\x03"):
                        self.running = False
                        break
                self._status()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self):
        self.init_dds()
        self.init_kinematics()

        # 1) 先确认相机流通了；收不到图就安全退出，绝不接管手臂
        self.rs = ZmqCamera(self.args.host, self.args.port, self.args.intrinsics)
        print(f"检测相机流 (最多等 {self.args.cam_timeout:.0f}s) ...", flush=True)
        if not self.rs.wait_for_stream(self.args.cam_timeout):
            self.rs.stop()
            print("ERROR: 未收到图像流。请确认 Jetson 上 image_server.py 已启动、"
                  "且 --host/--port 正确。未接管手臂，安全退出。")
            return
        print("相机流正常。")

        intr = self.rs.intrinsics()
        if intr is not None:
            with open(os.path.join(self.args.out, "intrinsics.json"), "w") as f:
                json.dump(intr, f, indent=2)
        else:
            print("WARNING: 未提供 --intrinsics，未写 intrinsics.json。"
                  "solve_handeye 需要它——请在 Jetson 用 dump_intrinsics.py 按推流分辨率"
                  "导出后补到 out 目录。")

        # 2) 手动确认后才接管左臂（进入软模式）
        try:
            input("\n相机已连通。请把手放到左臂旁准备扶住，按 Enter 开始接管左臂"
                  "（进入重力补偿软模式），Ctrl-C 取消...")
        except KeyboardInterrupt:
            self.rs.stop()
            print("\n已取消，未接管手臂。")
            return

        self.ctrl = RecurrentThread(interval=CONTROL_DT, target=self.control_step, name="arm_teach")
        self.ctrl.Start()
        print(f"Engaging arm_sdk... holding for {T_ENGAGE}s then softening left arm over "
              f"{T_RELEASE}s.\nSUPPORT the left arm now. Then move it by hand; "
              f"press c to capture, x to quit.\n")
        try:
            if self.args.no_preview:
                self.keyboard_loop()      # 无显示器时的纯键盘回退模式
            else:
                self.preview_loop()       # 默认: 实时预览窗口, 看图再决定拍照
        finally:
            self.running = False
            # 平滑交还上半身(weight 渐出), 防止退出瞬间左臂抖动/弹起
            try:
                self.release()
            except Exception as e:
                print(f"\n[release] 交还时出现异常(忽略): {e}")
            self.rs.stop()
            print(f"\nDone. {len(self.records)} samples in {self.args.out}")


if __name__ == "__main__":
    _DEFAULT_URDF = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "g1_description", "g1_29dof_with_hand_rev_1_0.urdf")) # update to use new urdf whose mode is 5

    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="enp12s0", help="上位机连机器人的网卡, 如 enp12s0")
    ap.add_argument("--urdf", default=_DEFAULT_URDF,
                    help="G1 29-DoF URDF (默认: calibration/unitree_g1_urdf/g1_29dof_with_hand.urdf)")
    ap.add_argument("--ee-frame", default="left_hand_palm_link")
    ap.add_argument("--out", default="./calib_data_" + time.strftime("%Y%m%d_%H%M"),
                    help="输出目录, 默认 ./calib_data_<日期_时分> (如 calib_data_20260615_0351)")
    ap.add_argument("--host", default="192.168.123.164", help="Jetson(image_server) IP")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--intrinsics", default=None,
                    help="相机内参 json (须与 image_server 推流分辨率一致)")
    ap.add_argument("--cam-timeout", type=float, default=8.0,
                    help="启动时等待相机图像流的超时(s); 超时则不接管手臂直接退出")
    ap.add_argument("--no-preview", action="store_true",
                    help="不开实时预览窗口, 回退到纯键盘模式(无显示器/SSH 无 X 时用)")
    args = ap.parse_args()

    print("WARNING: clear the workspace; keep a hand ready to support the LEFT arm.")
    input("Robot must be standing (锁定站立), motion mode NOT released. Press Enter...")
    ChannelFactoryInitialize(0, args.net)
    TeachCaptureRemote(args).run()
