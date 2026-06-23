"""读取 Unitree G1 的关节状态 (joint states)。

只订阅 rt/lowstate，不发任何指令 —— 纯只读，安全。
打印每个电机的角度 q、角速度 dq、估计力矩 tau_est，以及 IMU 姿态。

用法:
    python get_joint_states.py [网卡名]
例如:
    python get_joint_states.py enp12s0

不传网卡名时默认使用 enp12s0。

运行后会持续打印关节状态；按回车键即可把当前关节状态保存为 JSON 文件，
文件以时间戳命名，保存在 joint_states_record/ 文件夹下。
"""

import json
import os
import sys
import threading
import time
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# 关节状态记录的保存目录（位于项目根目录下）
RECORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "joint_states_record",
)

# G1 (humanoid) 29 自由度的关节名，索引与 lowstate.motor_state 一一对应
G1_JOINT_NAMES = [
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee", "LeftAnklePitch", "LeftAnkleRoll",
    "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee", "RightAnklePitch", "RightAnkleRoll",
    "WaistYaw", "WaistRoll", "WaistPitch",
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
    "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWristYaw",
]

G1_NUM_MOTOR = len(G1_JOINT_NAMES)


class JointStateReader:
    def __init__(self):
        self.low_state = None

    def init(self):
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg

    def print_once(self):
        s = self.low_state
        if s is None:
            print("尚未收到 lowstate，请确认网卡和机器人连接...")
            return
        print("=" * 78)
        rpy = s.imu_state.rpy
        print(f"IMU rpy (rad): roll={rpy[0]:+.3f}  pitch={rpy[1]:+.3f}  yaw={rpy[2]:+.3f}")
        print(f"{'idx':>3}  {'joint':<18} {'q(rad)':>9} {'dq(rad/s)':>10} {'tau_est(Nm)':>12}")
        for i in range(G1_NUM_MOTOR):
            m = s.motor_state[i]
            print(f"{i:>3}  {G1_JOINT_NAMES[i]:<18} {m.q:>9.3f} {m.dq:>10.3f} {m.tau_est:>12.3f}")

    def save_once(self):
        """把当前关节状态保存为时间戳命名的 JSON 文件，返回文件路径或 None。"""
        s = self.low_state
        if s is None:
            print("尚未收到 lowstate，无法保存。请确认网卡和机器人连接...")
            return None

        rpy = s.imu_state.rpy
        record = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "imu_rpy": {
                "roll": float(rpy[0]),
                "pitch": float(rpy[1]),
                "yaw": float(rpy[2]),
            },
            "joints": [],
        }
        for i in range(G1_NUM_MOTOR):
            m = s.motor_state[i]
            record["joints"].append({
                "idx": i,
                "name": G1_JOINT_NAMES[i],
                "q": float(m.q),
                "dq": float(m.dq),
                "tau_est": float(m.tau_est),
            })

        os.makedirs(RECORD_DIR, exist_ok=True)
        path = os.path.join(RECORD_DIR, f"{record['timestamp']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"已保存当前关节状态 -> {path}")
        return path


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else "enp12s0"
    print(f"使用网卡: {iface}  (订阅 rt/lowstate, 只读)")
    print("提示: 按回车键保存当前关节状态到 joint_states_record/，Ctrl+C 退出。")
    ChannelFactoryInitialize(0, iface)

    reader = JointStateReader()
    reader.init()

    # 后台线程监听回车，按下时保存当前关节状态
    def _input_loop():
        while True:
            try:
                input()
            except EOFError:
                break
            reader.save_once()

    threading.Thread(target=_input_loop, daemon=True).start()

    try:
        while True:
            time.sleep(0.5)
            reader.print_once()
    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
