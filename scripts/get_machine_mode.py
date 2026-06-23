"""读取 Unitree G1 的 machine mode (mode_machine)。

只订阅 rt/lowstate，不发任何指令 —— 纯只读，安全。
打印 lowstate 里的 mode_machine 和 mode_pr 两个字段：
  - mode_machine: 机型/固件标识 (底层控制发 lowcmd 时需要回填这个值)
  - mode_pr:      关节控制模式  0=PR(Pitch/Roll 串联)  1=AB(并联)

用法:
    python get_machine_mode.py [网卡名]
例如:
    python get_machine_mode.py enp12s0

不传网卡名时默认使用 enp12s0。
"""

import sys
import time

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

MODE_PR_NAMES = {0: "PR (Pitch/Roll 串联)", 1: "AB (并联)"}


class MachineModeReader:
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
        pr_name = MODE_PR_NAMES.get(s.mode_pr, "未知")
        print(f"mode_machine = {s.mode_machine}    mode_pr = {s.mode_pr} ({pr_name})    tick = {s.tick}")


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else "enp12s0"
    print(f"使用网卡: {iface}  (订阅 rt/lowstate, 只读)")
    ChannelFactoryInitialize(0, iface)

    reader = MachineModeReader()
    reader.init()

    try:
        while True:
            time.sleep(0.5)
            reader.print_once()
    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
