# 1. 连接机器人

## 设置网络环境
```bash
sudo ip addr add 192.168.123.99/24 dev enp12s0 && \
sudo ip link set enp12s0 up && \
for ip in 161 18 15 10; do ping -c 1 -W 1 192.168.123.$ip >/dev/null 2>&1 && echo "✅ 机器人在 192.168.123.$ip"; done
```

也可以手动设置：
- IP：192.168.123.99
- 子网掩码：255.255.255.0


## 连接unitree
```bash
ssh unitree@192.168.123.164
123
```

机器人网段设备一览：
- `192.168.123.161`：运动控制主控（只响应 ping，无 SSH，SDK 的 DDS 通信走这里）
- `192.168.123.164`：开发计算单元 Jetson Orin NX（SSH 登录用这个，用户名 `unitree`，密码 `123`）
- `192.168.123.99`：你的电脑（网卡 `enp12s0`）


# 2. 控制 G1（get / set joint states）

G1 是人形机器人（humanoid），使用 `unitree_hg` 消息，29 个自由度。控制分两层：
- 高层：走路/站立/挥手等封装动作（`LocoClient`）
- 底层：直接读写关节，订阅 `rt/lowstate` 拿状态、发布 `rt/lowcmd` 下指令

## 环境准备（已配好）
```bash
conda activate g1            # python 3.10 + unitree_sdk2py
```
SDK 源码在 `unitree_sdk2_python/`（以 editable 方式安装）。

## 读取关节状态（只读，安全）
```bash
conda activate g1
python get_joint_states.py enp12s0
```
打印每个电机的 q(角度) / dq(角速度) / tau_est(力矩) 和 IMU 姿态。

## 设置关节（底层控制，危险！）
⚠️ 第一次运行务必把机器人吊起来、双脚离地、周围无障碍，随时准备急停。
```bash
conda activate g1
# 仅保持当前姿态（最安全）
python set_joint_states.py enp12s0
# 让 18 号关节(LeftElbow) 在 3 秒内平滑移动到 0.5 rad
python set_joint_states.py enp12s0 --joint 18 --target 0.5 --move-time 3
```
脚本会先用 `MotionSwitcherClient` 释放高层运动模式，再以 500Hz 发送 `rt/lowcmd`。
关节索引见 `get_joint_states.py` 里的 `G1_JOINT_NAMES`。


# 3. 在上位机获取头部 RealSense D435i 图像

D435i 通过 USB 接在 Jetson(`192.168.123.164`) 上，上位机无法直接走 USB，
需要在 Jetson 上读图后通过网络(ZMQ)推给上位机。

- `image_server.py`：**在机器人 Jetson 上跑**，读 D435i → ZMQ 推流
- `image_client.py`：**在上位机跑**，接收并显示彩色 + 深度（鼠标悬停看距离）

## (1) 机器人端安装 pyrealsense2（一次性）
Jetson(aarch64) 上 `pip install pyrealsense2` 不可用，用 conda-forge 装：
```bash
# 在机器人 192.168.123.164 上
conda create -y -n cam -c conda-forge python=3.10 numpy py-opencv pyzmq
conda activate cam
pip install pyrealsense2
python -c "import pyrealsense2 as rs; print('pyrealsense2 OK')"
```
> 需要机器人能联网（conda-forge）。若机器人无外网，可用上位机做网络共享，或离线拷贝包。

Jetson上装 udev 规则
```bash
cd ~
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```
最后跑：`rs-enumerate-devices -s      # 现在应该能看到 D435I 了` 确认成功。

## (2) 把推流脚本拷到机器人
```bash
# 在上位机
scp ~/unitree_utils/image_server.py unitree@192.168.123.164:~/
```

## (3) 启动（两个终端）
机器人端：
```bash
ssh unitree@192.168.123.164
conda activate cam
python ~/image_server.py            # 彩色+深度 640x480@30, 端口 5555
```
上位机端：
```bash
conda activate g1
python ~/unitree_utils/image_client.py --host 192.168.123.164
```
窗口里按 `q` 退出。带宽紧张时机器人端加 `--no-depth` 或调小分辨率/`--jpeg-quality`。

