"""
G1 左臂【关节回放 + 拍照对比】—— 上位机版（相机走 ZMQ；独立脚本）。

配套 `1.g1_left_arm_teach_capture_remote.py`：方法一用拖动示教采了一批
<时间戳>.png(图片) + txt/<时间戳>.txt(含左臂关节角)。本脚本把这些关节角【慢速、
安全地】回放到机器人左臂，每到一个姿态拍一张图，存回同一文件夹：
    <时间戳>_replay.png   回放时拍的实际画面
    <时间戳>_diff.png     原图 | 回放图 | 差异热力图  三联对比

控制思路（与方法一一致，仅左臂由"软拖动"改为"慢速位置跟踪"）
----------------------------------------------------------------
- 通过 `rt/arm_sdk` 叠加接管上半身；腿不发指令，继续由锁定站立控制器管。
- 腰(12-14) + 右臂(22-28)：大刚度死锁在开机角度。
- 左臂(15-21)：位置控制(kp/kd 刚性)，指令角以【限速插值】缓慢逼近目标角，
  保证移动速度不会太快，安全。
- arm_sdk weight 0->1 渐入接管，退出时 1->0 渐出交还，避免抖动。

安全 / 前置条件（同方法一）
--------------------------
1. 机器人锁定站立 (L2 + Up)，不要释放运动模式。
2. 左手上仍固定着标定板(或回放时左臂工作空间清空、随时可扶)。
3. Jetson 端先启动 image_server.py 推流。
4. 启动时左臂会从【当前姿态】慢速移到【第一条记录的姿态】，请清空工作区、
   备好随时按 Ctrl-C / x 急停。

用法
----
python3 2.replay_and_compare.py ./calib_data_20260615_0416
python3 2.replay_and_compare.py ./calib_data_20260615_0416 --speed 0.2 --settle 1.5
"""

import argparse
import glob
import json
import os
import threading
import time

import numpy as np
import cv2
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

# 与方法一保持一致的 import：把采集脚本里的常量/相机类直接复用，避免重复维护。
# 文件名以数字开头不是合法模块名，用 importlib 按路径加载。
import importlib.util as _ilu

_CAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "1.g1_left_arm_teach_capture_remote.py")
_spec = _ilu.spec_from_file_location("g1_capture_remote", _CAP_PATH)
_cap = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cap)

# 复用方法一的常量/相机类/手臂配置（“能继承就不修改原代码”）
ZmqCamera = _cap.ZmqCamera
arm_joints = _cap.arm_joints              # side('left'/'right') -> {关节名: sdk idx}
held_sdk_idx = _cap.held_sdk_idx          # side -> 死锁(腰 + 另一条臂) 的 sdk idx
ALL_SDK = _cap.ALL_SDK                    # 全身可控关节 name -> sdk idx (建 FK 用)
WEIGHT_IDX = _cap.WEIGHT_IDX
CONTROL_DT = _cap.CONTROL_DT              # 100 Hz
KP_HOLD = _cap.KP_HOLD
KD_HOLD = _cap.KD_HOLD
T_ENGAGE = _cap.T_ENGAGE                  # arm_sdk weight 渐入/渐出时间

# 回放参数（默认值，命令行可覆盖）
DEFAULT_SPEED = 0.25     # rad/s, 每个左臂关节的最大角速度（限速，越小越慢越安全）
DEFAULT_SETTLE = 1.0     # s, 到位后稳定再拍照的最长等待时间
REACH_TOL = 5e-3         # rad, 指令角与目标角的到位阈值
TAU_CLAMP = 25.0         # Nm, 重力补偿前馈力矩限幅（安全）


class ReplayCompare:
    def __init__(self, args):
        self.args = args
        # 被回放臂(left/right): 默认按 --arm; 没指定则在 _list_records 里从数据自动识别
        self.arm = getattr(args, "arm", None) or "left"
        self.arm_joints = arm_joints(self.arm)
        self.held_idx = held_sdk_idx(self.arm)
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.weight = 0.0
        self.releasing = False
        self.lock = threading.Lock()
        # 被回放臂指令角(cmd_arm) 与 目标角(target_arm)，单位 rad，按 sdk idx 索引
        self.cmd_arm = {}
        self.target_arm = {}
        self.max_step = args.speed * CONTROL_DT     # 每控制周期允许的最大角度变化

    def _set_arm(self, side):
        """确定被回放臂并刷新关节/死锁配置 (回放必须在 init_targets 之前调用)。"""
        self.arm = side
        self.arm_joints = arm_joints(side)
        self.held_idx = held_sdk_idx(side)

    def _zero_waist(self):
        """把腰部 3 关节(yaw/roll/pitch = rpy)的死锁保持目标设为 0,0,0。
        腰部本就在死锁集里(大刚度保持), 这里只是把目标从当前角改成 0;
        接管时 arm_sdk weight 0->1 渐入, 腰会缓慢回正, 不会突然急动。
        需在 init_targets 之后(self.held 已建立)调用。"""
        for sdk in _cap.WAIST_SDK_IDX:
            if sdk in self.held:
                self.held[sdk] = 0.0
        print("腰部保持目标已设为 rpy=0,0,0 (接管后随 weight 渐入缓慢回正)。")

    # ---- DDS（同方法一）------------------------------------------------------
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

    def init_targets(self):
        """初始化锁定关节角 + 左臂指令/目标角(都=当前实测，先原地保持)。"""
        self.held = {sdk: self.low_state.motor_state[sdk].q for sdk in self.held_idx}
        for sdk in self.arm_joints.values():
            q = self.low_state.motor_state[sdk].q
            self.cmd_arm[sdk] = q
            self.target_arm[sdk] = q

    # ---- 运动学 / 重力补偿（同方法一建模, 用于前馈抵消重力下垂）-----------------
    def init_kinematics(self):
        if not self.args.gravcomp:
            self.arm_iv = None
            return
        try:
            self.model = pin.buildModelFromUrdf(self.args.urdf)  # 固定基座 @ pelvis
            self.fk_data = self.model.createData()
            # name -> idx_q (建全身配置), 左臂 name -> idx_v (取重力力矩)
            self.qmap = [(self.model.joints[self.model.getJointId(n)].idx_q, sdk)
                         for n, sdk in ALL_SDK.items() if n in self.model.names]
            self.arm_iv = [(self.model.joints[self.model.getJointId(n)].idx_v, sdk)
                            for n, sdk in self.arm_joints.items() if n in self.model.names]
            print(f"[gravcomp] 已加载 URDF 做重力补偿: {self.args.urdf}")
        except Exception as e:
            print(f"[gravcomp] 加载 URDF 失败, 关闭重力补偿: {e}")
            self.arm_iv = None

    def _full_q(self):
        q = pin.neutral(self.model)
        for idx_q, sdk in self.qmap:
            q[idx_q] = self.low_state.motor_state[sdk].q
        return q

    def _gravity_tau(self):
        """返回 {sdk: tau_ff}，抵消当前姿态下左臂各关节的重力力矩。"""
        if self.arm_iv is None:
            return {}
        g = pin.computeGeneralizedGravity(self.model, self.fk_data, self._full_q())
        scale = self.args.gravity_scale
        return {sdk: float(np.clip(g[idx_v] * scale, -TAU_CLAMP, TAU_CLAMP))
                for idx_v, sdk in self.arm_iv}

    # ---- 控制环 (100 Hz) -----------------------------------------------------
    def control_step(self):
        if self.low_state is None:
            return
        # weight: 接管渐入 / 退出渐出
        if self.releasing:
            self.weight = max(0.0, self.weight - CONTROL_DT / T_ENGAGE)
        else:
            self.weight = min(1.0, self.weight + CONTROL_DT / T_ENGAGE)
        self.low_cmd.motor_cmd[WEIGHT_IDX].q = self.weight

        # 重力补偿前馈力矩(随当前实测姿态实时计算), 抵消左臂因 kp 偏软的重力下垂
        gtau = self._gravity_tau()

        # 左臂：限速插值逼近目标角，刚性位置控制 + 重力补偿
        with self.lock:
            for sdk in self.arm_joints.values():
                cur = self.cmd_arm[sdk]
                tgt = self.target_arm[sdk]
                d = tgt - cur
                if abs(d) > self.max_step:
                    cur += self.max_step if d > 0 else -self.max_step
                else:
                    cur = tgt
                self.cmd_arm[sdk] = cur
                m = self.low_cmd.motor_cmd[sdk]
                m.q, m.dq, m.tau = float(cur), 0.0, gtau.get(sdk, 0.0)
                m.kp, m.kd = KP_HOLD, KD_HOLD

        # 腰 + 右臂：死锁
        for sdk, qh in self.held.items():
            m = self.low_cmd.motor_cmd[sdk]
            m.q, m.dq, m.tau, m.kp, m.kd = float(qh), 0.0, 0.0, KP_HOLD, KD_HOLD

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _set_target(self, names, q):
        with self.lock:
            for n, val in zip(names, q):
                if n in self.arm_joints:
                    self.target_arm[self.arm_joints[n]] = float(val)

    def _reached(self):
        with self.lock:
            return all(abs(self.target_arm[s] - self.cmd_arm[s]) <= REACH_TOL
                       for s in self.arm_joints.values())

    def _max_move_dist(self, names, q):
        """目标相对当前指令角的最大单关节移动量(rad)，用于估算到位时间。"""
        with self.lock:
            d = 0.0
            for n, val in zip(names, q):
                if n in self.arm_joints:
                    d = max(d, abs(float(val) - self.cmd_arm[self.arm_joints[n]]))
            return d

    def wait_until_reached(self, est_time):
        """阻塞等左臂指令角到位（限速插值完成）；超时按估算时间 + 余量保护。"""
        deadline = time.time() + est_time + 3.0
        while not self._reached():
            if time.time() > deadline:
                print("\n[warn] 到位超时，继续（请检查左臂是否被卡住）")
                return
            time.sleep(0.05)

    def _measured_err_deg(self, names, q):
        """实测左臂关节角与目标角的最大单关节误差(度)，验证是否真正到位。"""
        e = 0.0
        for n, val in zip(names, q):
            if n in self.arm_joints:
                m = self.low_state.motor_state[self.arm_joints[n]].q
                e = max(e, abs(m - float(val)))
        return float(np.degrees(e))

    def wait_settled(self, names, q):
        """指令到位后再等【实测角】稳定到目标附近(或超时), 返回最终实测误差(度)。"""
        deadline = time.time() + self.args.settle + 3.0
        while time.time() < deadline:
            if self._measured_err_deg(names, q) <= self.args.tol_deg:
                break
            time.sleep(0.05)
        time.sleep(0.3)   # 相机/画面稳定
        return self._measured_err_deg(names, q)

    # ---- debug 显示 ----------------------------------------------------------
    @staticmethod
    def _short(name):
        return _cap._short_joint(name)

    def _measured_q_list(self, names):
        return [self.low_state.motor_state[self.arm_joints[n]].q
                for n in names if n in self.arm_joints]

    def _fmt_q_deg(self, names, q):
        return "  ".join(f"{self._short(n)}={np.degrees(v):+6.1f}"
                         for n, v in zip(names, q) if n in self.arm_joints)

    def debug_drive(self, names, q, orig_img):
        """debug 模式: 边等到位边显示【实时画面/采集原图】两窗口, 滚动打印当前关节角。
        到位且实测误差达标后返回最终误差(度); 窗口里按 x/q 可中断。"""
        deadline = time.time() + self.args.settle + 6.0
        settled_since = None
        last = None
        while time.time() < deadline:
            frame = self.rs.capture(timeout_ms=100)
            if frame is not None:
                last = frame
            if last is not None:
                cv2.imshow("replay-live", last)
            if orig_img is not None:
                cv2.imshow("recorded-original", orig_img)
            if last is not None and orig_img is not None:
                heat, score = self._diff_heat(orig_img, last)
                cv2.imshow("diff (live vs recorded)",
                           self._label(heat, f"diff mean={score:.1f}"))

            cur = self._measured_q_list(names)
            err = self._measured_err_deg(names, q)
            print(f"[cur] {self._fmt_q_deg(names, cur)}  | max_err={err:5.2f}deg   ",
                  end="\r", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("x"), ord("q"), 27):
                raise KeyboardInterrupt

            if self._reached() and err <= self.args.tol_deg:
                if settled_since is None:
                    settled_since = time.time()
                elif time.time() - settled_since >= 0.3:
                    break
            else:
                settled_since = None
        print()   # 结束滚动行
        return self._measured_err_deg(names, q)

    def release(self):
        """退出：左臂保持当前姿态，arm_sdk weight 渐出交还站立控制器。"""
        if self.low_state is None:
            return
        with self.lock:
            for sdk in self.arm_joints.values():
                self.target_arm[sdk] = self.low_state.motor_state[sdk].q
                self.cmd_arm[sdk] = self.target_arm[sdk]
        self.releasing = True
        print("\n交还上半身给站立控制器 (weight -> 0) ...")
        time.sleep(T_ENGAGE + 0.5)

    # ---- 差异图 --------------------------------------------------------------
    @staticmethod
    def _label(img, text):
        cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _diff_heat(orig, replay):
        """返回 (差异热力图, 平均差异); 分辨率不一致时把原图缩放到 replay 尺寸。"""
        if orig.shape != replay.shape:
            orig = cv2.resize(orig, (replay.shape[1], replay.shape[0]))
        gray = cv2.cvtColor(cv2.absdiff(orig, replay), cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET), float(gray.mean())

    def make_diff(self, orig, replay):
        """生成三联对比图 [原图 | 回放图 | 差异热力图]，返回 (图, 平均差异)。"""
        if orig.shape != replay.shape:
            orig = cv2.resize(orig, (replay.shape[1], replay.shape[0]))
        heat, score = self._diff_heat(orig, replay)
        a = self._label(orig.copy(), "original")
        b = self._label(replay.copy(), "replay")
        c = self._label(heat.copy(), f"diff (mean={score:.1f})")
        return np.hstack([a, b, c]), score

    # ---- 主流程 --------------------------------------------------------------
    def _list_records(self):
        """收集数据目录 txt/ 下的采样记录（png 在根目录, 与 txt 同名时间戳）。
        顺便从 txt 的 arm 字段识别本批数据用的是哪条臂, 存到 self._data_arm。"""
        recs = []
        arms = set()
        txt_dir = os.path.join(self.args.data, "txt")
        for tp in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
            stamp = os.path.splitext(os.path.basename(tp))[0]
            try:
                r = _cap.parse_capture_txt(tp)
            except Exception as e:
                print(f"[skip] 读取失败 {os.path.basename(tp)}: {e}")
                continue
            if not r["joint_names"] or not r["joint_q"]:
                continue
            arms.add(r.get("arm", "left"))
            recs.append((stamp, stamp + ".png", r["joint_names"], r["joint_q"]))
        if len(arms) > 1:
            print(f"WARNING: 数据里混了多条臂 {arms}, 将以 {sorted(arms)[0]} 为准(可用 --arm 覆盖)。")
        self._data_arm = (sorted(arms)[0] if arms else "left")
        return recs

    def run(self):
        self.init_dds()

        records = self._list_records()
        if not records:
            print(f"ERROR: 在 {self.args.data}/txt 没找到可回放的记录 txt。")
            return
        # 优先用命令行 --arm, 否则用数据里识别出的臂
        self._set_arm(self.args.arm or self._data_arm)
        print(f"找到 {len(records)} 条记录待回放。(臂: {self.arm})")

        self.init_targets()
        self.init_kinematics()

        # 相机流先探活，收不到图就退出，不接管手臂
        self.rs = ZmqCamera(self.args.host, self.args.port, self.args.intrinsics)
        print(f"检测相机流 (最多等 {self.args.cam_timeout:.0f}s) ...", flush=True)
        if not self.rs.wait_for_stream(self.args.cam_timeout):
            self.rs.stop()
            print("ERROR: 未收到图像流。请确认 Jetson 上 image_server.py 已启动、"
                  "且 --host/--port 正确。未接管手臂，安全退出。")
            return
        print("相机流正常。")

        # 图片传输检测通过后: 把腰部 rpy 设为 0,0,0 (接管后随 weight 渐入回正)
        self._zero_waist()

        arm_cn = "左臂" if self.arm == "left" else "右臂"
        try:
            input(f"\n相机已连通。将以 {self.args.speed} rad/s 慢速回放 {len(records)} 个姿态。\n"
                  f"请清空{arm_cn}工作区、备好随时急停(Ctrl-C / x)，按 Enter 开始接管{arm_cn}...")
        except KeyboardInterrupt:
            self.rs.stop()
            print("\n已取消，未接管手臂。")
            return

        self.ctrl = RecurrentThread(interval=CONTROL_DT, target=self.control_step, name="arm_replay")
        self.ctrl.Start()
        print(f"Engaging arm_sdk (weight 0->1, {T_ENGAGE}s) ... {self.arm} 臂将开始慢速移动。\n")
        time.sleep(T_ENGAGE + 0.3)   # 等 weight 接管到位再开始动

        if self.args.debug:
            cv2.namedWindow("replay-live", cv2.WINDOW_NORMAL)
            cv2.namedWindow("recorded-original", cv2.WINDOW_NORMAL)
            cv2.namedWindow("diff (live vs recorded)", cv2.WINDOW_NORMAL)

        done = 0
        try:
            for i, (stamp, img_name, names, q) in enumerate(records):
                dist = self._max_move_dist(names, q)
                est = dist / max(self.args.speed, 1e-6)
                print(f"[{i+1}/{len(records)}] -> {stamp}  "
                      f"max_move={np.degrees(dist):.1f}deg  est={est:.1f}s")

                # 预读采集原图(debug 显示 + 后面做 diff 复用)
                orig_path = os.path.join(self.args.data, img_name)
                orig = cv2.imread(orig_path) if os.path.isfile(orig_path) else None

                if self.args.debug:
                    print(f"  [record joint] {self._fmt_q_deg(names, q)}")

                self._set_target(names, q)
                if self.args.debug:
                    err_deg = self.debug_drive(names, q, orig)   # 显示双窗口+滚动打印
                else:
                    self.wait_until_reached(est)
                    err_deg = self.wait_settled(names, q)        # 等实测角真正到位再拍
                tag = "" if err_deg <= self.args.tol_deg else "  [warn 关节未到位!]"
                print(f"  joint_err={err_deg:.2f}deg{tag}")

                frame = self.rs.capture(timeout_ms=2000)
                if frame is None:
                    print("  [skip] 没收到相机帧，跳过此姿态拍照。")
                    continue

                replay_path = os.path.join(self.args.data, stamp + "_replay.png")
                cv2.imwrite(replay_path, frame)

                if orig is not None:
                    comp, score = self.make_diff(orig, frame)
                    diff_path = os.path.join(self.args.data, stamp + "_diff.png")
                    cv2.imwrite(diff_path, comp)
                    print(f"  saved {os.path.basename(replay_path)} + "
                          f"{os.path.basename(diff_path)}  diff_mean={score:.2f}")
                else:
                    print(f"  [warn] 原图不存在 {img_name}，只存了 _replay.png（无 diff）。")
                done += 1
        except KeyboardInterrupt:
            print("\n收到中断，停止回放。")
        finally:
            if self.args.debug:
                cv2.destroyAllWindows()
            try:
                self.release()
            except Exception as e:
                print(f"\n[release] 交还时异常(忽略): {e}")
            self.rs.stop()
            print(f"\nDone. 回放并对比 {done}/{len(records)} 个姿态，输出在 {self.args.data}")


if __name__ == "__main__":
    _DEFAULT_URDF = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "g1_description", "g1_29dof_with_hand_rev_1_0.urdf"))

    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="采集数据目录(含 <时间戳>.png + txt/<时间戳>.txt)，如 ./calib_data_20260615_0416")
    ap.add_argument("--net", default="enp12s0", help="上位机连机器人的网卡, 如 enp12s0")
    ap.add_argument("--arm", choices=["left", "right"], default=None,
                    help="回放哪条臂; 默认按数据 txt 里的 arm 字段自动识别")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help=f"左臂每关节最大角速度 rad/s (越小越慢越安全, 默认 {DEFAULT_SPEED})")
    ap.add_argument("--settle", type=float, default=DEFAULT_SETTLE,
                    help=f"到位后等实测角稳定的最长秒数 (默认 {DEFAULT_SETTLE})")
    ap.add_argument("--tol-deg", type=float, default=1.5,
                    help="实测关节角到位阈值(度), 超过会打 warn (默认 1.5)")
    ap.add_argument("--urdf", default=_DEFAULT_URDF, help="G1 29-DoF URDF (重力补偿用)")
    ap.add_argument("--no-gravcomp", dest="gravcomp", action="store_false",
                    help="关闭重力补偿前馈(默认开启; 关闭后左臂会因 kp 偏软而下垂)")
    ap.add_argument("--gravity-scale", type=float, default=1.0,
                    help="重力补偿力矩缩放(默认 1.0; 若仍下垂可略调大, 抖动则调小)")
    ap.add_argument("--host", default="192.168.123.164", help="Jetson(image_server) IP")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--intrinsics", default=None, help="相机内参 json (可选, 仅用于相机类初始化)")
    ap.add_argument("--cam-timeout", type=float, default=8.0,
                    help="启动时等待相机图像流的超时(s); 超时则不接管手臂直接退出")
    ap.add_argument("--debug", action="store_true",
                    help="调试模式: 开两个窗口(实时画面/采集原图)肉眼比对, "
                         "终端打印 record 关节角并滚动打印当前关节角(窗口内 x/q 中断)")
    ap.set_defaults(gravcomp=True)
    args = ap.parse_args()

    _arm_hint = args.arm or "数据自动识别"
    print(f"WARNING: clear the workspace; keep a hand ready to support the arm ({_arm_hint}).")
    print(f"将慢速回放 {os.path.abspath(args.data)} 里的关节姿态 (臂: {_arm_hint})。")
    input("Robot must be standing (锁定站立), motion mode NOT released. Press Enter...")
    ChannelFactoryInitialize(0, args.net)
    ReplayCompare(args).run()
