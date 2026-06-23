"""让 Unitree G1 的【左臂进入阻尼模式】, 其余关节维持锁定站立。

原理
----
G1 的 `rt/arm_sdk` 通道会把"手臂控制"叠加在板载站立/全身控制器之上:
  - 双腿: 始终由锁定站立控制器接管, 本脚本完全不碰;
  - 上半身(腰 + 双臂): 一旦置位 motor_cmd[29].q = weight 把 arm_sdk 打开, 就交给本脚本。

注意 arm_sdk 接管的是【整个上半身组】, 无法只单独接管左臂。所以本脚本:
  - 左臂(15-21): 阻尼模式  kp=0, kd>0, q=0, dq=0, tau=0  -> 可用手自由拖动, 带阻尼, 不回弹;
  - 右臂(22-28) + 腰(12-14): 锁在启动时的角度(正常 kp/kd 保持不动);
  - 双腿: 不发指令, 继续由锁定站立控制器管。

运行前
------
1. 机器人保持【锁定站立】(手柄 L2 + Up)。本脚本不会释放高层模式,
   arm_sdk 是叠加在站立控制器之上的。
2. 左臂下方/周围清空: 进入阻尼后左臂会因重力缓慢下垂, 随时准备扶住。
3. 退出(Ctrl-C)时脚本会把 weight 平滑降回 0, 让站立控制器重新接管上半身。

!!!!!!!!!!!!!!!!!!!!!!!!!!!  危  险  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
本脚本会真实驱动 G1 电机。第一次运行建议把机器人吊起来、双脚离地、随时准备急停。
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

用法:
    python g1_left_arm_damping.py [网卡名]
    python g1_left_arm_damping.py enp12s0 --damp-kd 2.0

关节索引见 get_joint_states.py 中的 G1_JOINT_NAMES。
"""

import sys
import time
import argparse

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

G1_NUM_MOTOR = 29

# 左臂 7 个关节: 进入阻尼模式
LEFT_ARM_JOINTS = list(range(15, 22))          # 15..21
# 由 arm_sdk 锁住保持不动的关节: 腰 + 右臂
HELD_JOINTS = list(range(12, 15)) + list(range(22, 29))  # 12..14, 22..28
WEIGHT_IDX = 29                                # arm_sdk 使能位 motor_cmd[29].q


class LeftArmDamping:
    def __init__(self, damp_kd=2.0, hold_kp=60.0, hold_kd=1.5, ramp_time=2.0):
        self.control_dt = 0.02       # 50 Hz, 与官方 arm_sdk 示例一致
        self.damp_kd = damp_kd       # 左臂阻尼系数 (kp=0)
        self.hold_kp = hold_kp       # 保持关节(腰+右臂)的刚度
        self.hold_kd = hold_kd       # 保持关节的阻尼
        self.ramp_time = ramp_time   # weight 渐入/渐出用时 (s)

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.mode_machine = 0
        self.got_state = False

        self.q_init = [0.0] * G1_NUM_MOTOR  # 启动时记录的角度(用于锁住保持关节)
        self.weight = 0.0
        self.releasing = False              # 退出时置位, weight 渐出

    def init(self):
        # 不释放高层模式: arm_sdk 叠加在锁定站立控制器之上。
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_low_state, 10)

        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        if not self.got_state:
            self.mode_machine = msg.mode_machine
            for i in range(G1_NUM_MOTOR):
                self.q_init[i] = msg.motor_state[i].q
            self.got_state = True

    def start(self):
        print("等待 lowstate ...")
        while not self.got_state:
            time.sleep(0.1)
        print("已获取当前姿态, 左臂进入阻尼模式 (Ctrl-C 退出并交还站立控制器)。")
        self.thread = RecurrentThread(interval=self.control_dt, target=self._write, name="control")
        self.thread.Start()

    def _write(self):
        # weight 渐入(0->1)启用 arm_sdk; 退出时渐出(1->0)交还站立控制器
        step = self.control_dt / self.ramp_time
        if self.releasing:
            self.weight = max(0.0, self.weight - step)
        else:
            self.weight = min(1.0, self.weight + step)

        self.low_cmd.mode_machine = self.mode_machine
        self.low_cmd.motor_cmd[WEIGHT_IDX].q = self.weight

        # 左臂: 阻尼模式 (kp=0, kd>0, 目标角速度 0) -> 带阻尼可自由拖动
        for j in LEFT_ARM_JOINTS:
            m = self.low_cmd.motor_cmd[j]
            m.mode = 1
            m.q = 0.0
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = self.damp_kd

        # 腰 + 右臂: 锁在启动角度
        for j in HELD_JOINTS:
            m = self.low_cmd.motor_cmd[j]
            m.mode = 1
            m.q = self.q_init[j]
            m.dq = 0.0
            m.tau = 0.0
            m.kp = self.hold_kp
            m.kd = self.hold_kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def release(self):
        """平滑把 weight 降回 0, 让站立控制器重新接管上半身。"""
        self.releasing = True
        # 等待 weight 渐出到 0 (留些余量)
        time.sleep(self.ramp_time + 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("iface", nargs="?", default="enp12s0", help="网卡名, 默认 enp12s0")
    parser.add_argument("--damp-kd", type=float, default=2.0, help="左臂阻尼系数 kd (越大越粘手), 默认 2.0")
    parser.add_argument("--hold-kp", type=float, default=60.0, help="腰+右臂保持刚度 kp, 默认 60")
    parser.add_argument("--hold-kd", type=float, default=1.5, help="腰+右臂保持阻尼 kd, 默认 1.5")
    parser.add_argument("--ramp-time", type=float, default=2.0, help="weight 渐入/渐出用时 (s), 默认 2.0")
    args = parser.parse_args()

    print("=" * 72)
    print("G1 左臂阻尼模式: 左臂可自由拖动, 腰+右臂锁定, 双腿继续锁定站立。")
    print("运行前请确认: 机器人处于【锁定站立】(L2+Up), 左臂下方清空, 准备扶住。")
    print(f"网卡={args.iface}  damp_kd={args.damp_kd}  hold_kp={args.hold_kp}  hold_kd={args.hold_kd}")
    print("=" * 72)
    input("确认安全后按 Enter 继续 (Ctrl-C 取消)...")

    ChannelFactoryInitialize(0, args.iface)

    ctrl = LeftArmDamping(
        damp_kd=args.damp_kd,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        ramp_time=args.ramp_time,
    )
    ctrl.init()
    ctrl.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在交还上半身给站立控制器 (weight -> 0) ...")
        ctrl.release()
        print("已退出。")


if __name__ == "__main__":
    main()
