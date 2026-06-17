"""
G1 左臂【多次回放 + 重复性/精度统计 + 误差绘图】—— 上位机版（独立脚本）。

在方法二 `2.replay_and_compare_with_record_results.py` 基础上：把同一批采集姿态
【慢速、安全地】回放 N 次(默认 3 次)，每次记录左臂【实测关节角】和【实测末端位姿
(FK)】，与采集时记录的目标值对比，统计：
- joint states 误差：每关节 / 整体的 mean / max / min（度）
- eef 误差：位置(mm) 和 姿态(deg) 的 mean / max
并画一张【每关节 mean/max/min】误差图。

输出目录结构（在采集数据目录下新建 replay/）
--------------------------------------------
<data>/replay/
    <run时间戳_1>/  <stamp>.png             # 回放时拍的画面
                    diff/<stamp>_diff.png   # 原图|回放|差异 三联对比(单独子文件夹)
                    <stamp>.txt             # 每姿态: 实测 + 目标 + 误差(measured/target/error)
                    <stamp>_all_joint.json  # 每姿态: 实测全身 29 关节状态 + IMU(同 get_joint_states 格式)
                    measured.json           # 左臂实测/目标/误差 json 汇总(保留, 不变)
    <run时间戳_2>/  ...
    <run时间戳_3>/  ...
    summary.json            # 全部统计汇总
    joint_error_stats.png   # 每关节 mean/max/min + eef 误差图

控制 / 安全：完全复用方法二（限速插值位置控制 + 重力补偿前馈 + 实测到位判定）。

用法
----
terminal 1:
- cd lancel/ && conda activate cam && python image_server.py 
terminal 2:
- cd /home/lancel/unitree_utils/workflow && conda activate g1 && python3 3.replay_and_compare_replay_results.py ./calib_data_20260615_0416 --runs 3 --speed 0.2
"""

import argparse
import importlib.util as _ilu
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import pinocchio as pin

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.thread import RecurrentThread


def _load_module(name, rel_path):
    """按文件路径加载模块（文件名以数字开头不是合法模块名）。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    spec = _ilu.spec_from_file_location(name, p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 复用方法二的运动控制（“能继承就不修改原代码”）
_m2 = _load_module("replay_compare_m2", "2.replay_and_compare_with_record_results.py")
ReplayCompare = _m2.ReplayCompare
ZmqCamera = _m2.ZmqCamera
arm_joints = _m2.arm_joints              # side -> {关节名: sdk idx}
CONTROL_DT = _m2.CONTROL_DT
T_ENGAGE = _m2.T_ENGAGE

# 复用参考脚本 2a 的棋盘格角点/EEF 稳定性算法（同样的对比方式, 不改原文件）
_stab = _load_module("stability_chessboard",
                     os.path.join("..", "reference_code",
                                  "2a.compute_replay_stablebility_chessBoard.py"))
# 复用 scripts/get_joint_states.py 的全身关节命名（all_joint.json 与其同格式）
_gjs = _load_module("get_joint_states",
                    os.path.join("..", "scripts", "get_joint_states.py"))
G1_ALL_JOINT_NAMES = _gjs.G1_JOINT_NAMES

EefPose = _stab.EefPose
detect_corners = _stab.detect_corners
corner_shift_stats = _stab.corner_shift_stats
eef_pairwise = _stab.eef_pairwise
save_corner_overlay = _stab.save_corner_overlay
_Tee = _stab._Tee
CORNER_GOOD_PX = _stab.CORNER_GOOD_PX
CORNER_OK_PX = _stab.CORNER_OK_PX
CORNER_WARN_PX = _stab.CORNER_WARN_PX
DEFAULT_PATTERN_COLS = _stab.DEFAULT_PATTERN_COLS
DEFAULT_PATTERN_ROWS = _stab.DEFAULT_PATTERN_ROWS
DEFAULT_N_WORST = _stab.DEFAULT_N_WORST

# 规范关节顺序(self.joint_names)在确定被回放臂后于 _load_records_full 里设置


def _rot_angle_deg(Ra, Rb):
    """两旋转矩阵之间的夹角(度)。"""
    rrel = np.asarray(Ra).T @ np.asarray(Rb)
    c = (np.trace(rrel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class ReplayMultiRun(ReplayCompare):
    """多次回放 + 统计。复用 ReplayCompare 的运动控制, 仅扩展 FK 末端 + 主流程。"""

    def init_kinematics(self):
        # 末端坐标系没显式指定时, 按被回放臂取 <arm>_wrist_yaw_link
        if not self.args.ee_frame:
            self.args.ee_frame = f"{self.arm}_wrist_yaw_link"
        # 强制开启建模（FK 比较必需），复用父类建 model/qmap/重力补偿
        self.args.gravcomp = True
        super().init_kinematics()
        if not hasattr(self, "model"):
            raise RuntimeError("URDF 未能加载, 无法做 FK 末端比较, 请检查 --urdf。")
        if not self.model.existFrame(self.args.ee_frame):
            raise RuntimeError(f"URDF 中找不到末端坐标系: {self.args.ee_frame}")
        self.ee_id = self.model.getFrameId(self.args.ee_frame)

    def _measured_ee(self):
        """当前实测姿态下的末端位姿 (pelvis 固定基座, 与采集时同一套 FK)。"""
        pin.framesForwardKinematics(self.model, self.fk_data, self._full_q())
        return self.fk_data.oMf[self.ee_id].copy()

    def _save_all_joint_json(self, run_dir, stamp):
        """把当前【实测】全身关节状态(29 关节 + IMU)存为 <stamp>_all_joint.json,
        字段格式与 scripts/get_joint_states.py 一致(timestamp + imu_rpy + joints)。"""
        s = self.low_state
        if s is None:
            return
        rpy = s.imu_state.rpy
        record = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "imu_rpy": {"roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])},
            "joints": [
                {"idx": i, "name": G1_ALL_JOINT_NAMES[i],
                 "q": float(s.motor_state[i].q),
                 "dq": float(s.motor_state[i].dq),
                 "tau_est": float(s.motor_state[i].tau_est)}
                for i in range(len(G1_ALL_JOINT_NAMES))
            ],
        }
        with open(os.path.join(run_dir, stamp + "_all_joint.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _diff_dir(run_dir):
        """对比图(_diff.png)的子目录: <run_dir>/diff/。回放图本身存在 run_dir 下 <stamp>.png。"""
        d = os.path.join(run_dir, "diff")
        os.makedirs(d, exist_ok=True)
        return d

    def _capture_fresh(self, flush_s=0.6):
        """拿一张【新鲜】帧。

        ZMQ SUB 有 RCVHWM 缓冲, 姿态间长时间不读会积压旧帧, 只调一次 capture()
        会拿到移动途中的旧画面("还没到位就拍照"的根因)。这里持续读 flush_s 秒,
        把积压旧帧冲掉, 始终保留最新一帧再返回。
        """
        if self.rs is None:
            return None
        end = time.time() + flush_s
        frame = None
        while time.time() < end:
            f = self.rs.capture(timeout_ms=100)
            if f is not None:
                frame = f
        return frame

    # ---- 记录加载（含 eef 目标）---------------------------------------------
    def _load_records_full(self):
        recs = []
        import glob
        txt_dir = os.path.join(self.args.data, "txt")
        parsed = []
        arms = set()
        for tp in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
            stamp = os.path.splitext(os.path.basename(tp))[0]
            try:
                r = _m2._cap.parse_capture_txt(tp)
            except Exception:
                continue
            if not r["joint_names"] or not r["joint_q"]:
                continue
            arms.add(r.get("arm", "left"))
            parsed.append((stamp, r))
        if not parsed:
            return recs
        if len(arms) > 1:
            print(f"WARNING: 数据里混了多条臂 {arms}, 将以 {sorted(arms)[0]} 为准(可用 --arm 覆盖)。")
        # 确定被回放臂 + 规范关节顺序(self.joint_names)
        self._set_arm(self.args.arm or sorted(arms)[0])
        self.joint_names = list(self.arm_joints.keys())
        for stamp, r in parsed:
            # 目标关节角对齐到规范顺序
            qmap = dict(zip(r["joint_names"], r["joint_q"]))
            q_tgt = [float(qmap.get(n, 0.0)) for n in self.joint_names]
            # 目标姿态: 由四元数重建旋转矩阵(txt 不存 R), 平移直接取
            R_tgt = _m2._cap.R_from_quat(r["quat_wxyz"])
            t_tgt = np.array(r["tcp_xyz"], float)
            recs.append(dict(stamp=stamp, img=stamp + ".png", q_tgt=q_tgt,
                             R_tgt=R_tgt, t_tgt=t_tgt))
        return recs

    # ---- 单次回放 ------------------------------------------------------------
    def _run_once(self, records, run_dir):
        """回放一遍所有姿态, 返回 per-pose 的实测/误差列表; 顺带存图。"""
        os.makedirs(run_dir, exist_ok=True)
        results = []
        for i, r in enumerate(records):
            q_tgt = r["q_tgt"]
            self._set_target(self.joint_names, q_tgt)
            # 每个姿态拍照/记录前都把腰部 3 关节(rpy)的保持目标重设为 0,0,0,
            # 与手臂运动同时回零, 等手臂稳定时腰部也已到位再拍照采数据。
            self._zero_waist()
            dist = self._max_move_dist(self.joint_names, q_tgt)
            est = dist / max(self.args.speed, 1e-6)
            self.wait_until_reached(est)
            self.wait_settled(self.joint_names, q_tgt)
            time.sleep(1.0)   # 到位后静置 1s, 等机械臂/画面完全稳定再拍照采数据

            q_meas = self._measured_q_list(self.joint_names)
            T = self._measured_ee()
            t_meas = np.asarray(T.translation, float)
            R_meas = np.asarray(T.rotation, float)

            j_err = [abs(m - t) for m, t in zip(q_meas, q_tgt)]      # rad, 规范顺序
            pos_err = float(np.linalg.norm(t_meas - r["t_tgt"]))     # m
            rot_err = _rot_angle_deg(r["R_tgt"], R_meas)             # deg

            print(f"    [{i+1}/{len(records)}] {r['stamp']}  "
                  f"j_max={np.degrees(max(j_err)):.2f}deg  "
                  f"pos={pos_err*1000:.1f}mm  rot={rot_err:.2f}deg")

            # 拍照存图（相机可用时）。用 _capture_fresh 冲掉积压旧帧, 确保是到位后的画面
            # 回放图存成 run_dir/<stamp>.png; 对比图存到子文件夹 diff/<stamp>_diff.png
            if self.rs is not None:
                frame = self._capture_fresh()
                if frame is not None:
                    cv2.imwrite(os.path.join(run_dir, r["stamp"] + ".png"), frame)
                    orig_path = os.path.join(self.args.data, r["img"])
                    if os.path.isfile(orig_path):
                        comp, _ = self.make_diff(cv2.imread(orig_path), frame)
                        cv2.imwrite(os.path.join(self._diff_dir(run_dir),
                                                 r["stamp"] + "_diff.png"), comp)

            # 额外存每姿态 txt(实测 + 目标 + 误差); measured.json 仍照常保存(不变)
            txt = _m2._cap.format_replay_txt(
                r["stamp"], self.joint_names,
                q_meas, t_meas.tolist(),
                _m2._cap.rotvec_from_R(R_meas).tolist(), _m2._cap.quat_wxyz_from_R(R_meas),
                q_tgt, r["t_tgt"].tolist(),
                _m2._cap.rotvec_from_R(r["R_tgt"]).tolist(), _m2._cap.quat_wxyz_from_R(r["R_tgt"]),
                j_err, pos_err, rot_err, arm=self.arm,
            )
            with open(os.path.join(run_dir, r["stamp"] + ".txt"), "w") as f:
                f.write(txt)

            # 额外存当前实测的全身关节状态(全 29 关节 + IMU)
            self._save_all_joint_json(run_dir, r["stamp"])

            results.append(dict(
                stamp=r["stamp"],
                q_meas=q_meas, q_tgt=q_tgt,
                joint_err_rad=j_err,
                t_meas=t_meas.tolist(), t_tgt=r["t_tgt"].tolist(),
                r_meas_rotvec=cv2.Rodrigues(R_meas)[0].flatten().tolist(),
                eef_pos_err_m=pos_err, eef_rot_err_deg=rot_err,
            ))

        with open(os.path.join(run_dir, "measured.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results

    # ---- 统计 + 绘图 ---------------------------------------------------------
    @staticmethod
    def _aggregate(all_runs):
        """all_runs: list[run] of list[pose] dict.
        汇总【多次回放之间】的重复性(两两差异), 而不是回放与采集记录的误差:
        - joint: 每姿态各次回放【实测角】两两差异(deg), 逐关节统计
        - eef  : 每姿态各次回放【实测末端位姿】两两差异(复用 2a 的 eef_pairwise)
        """
        n_runs = len(all_runs)
        n_poses = len(all_runs[0])

        # joint: 实测角 (R,P,J), 取各次回放两两差异(度)
        qm = np.array([[p["q_meas"] for p in run] for run in all_runs])  # (R,P,J) rad
        pair_idx = [(i, k) for i in range(n_runs) for k in range(i + 1, n_runs)]
        if pair_idx:
            jpair = np.array([np.abs(qm[i] - qm[k]) for i, k in pair_idx])  # (npair,P,J)
        else:
            jpair = np.zeros((1,) + qm.shape[1:])   # 只有 1 次回放: 无重复性, 记 0
        jpair_deg = np.degrees(jpair)
        flat = jpair_deg.reshape(-1, jpair_deg.shape[-1])      # (npair*P, J)
        joint = dict(
            mean=flat.mean(axis=0).tolist(),
            max=flat.max(axis=0).tolist(),
            min=flat.min(axis=0).tolist(),
            overall_mean=float(flat.mean()),
            overall_max=float(flat.max()),
            per_pose_mean=[float(jpair_deg[:, p, :].mean()) for p in range(n_poses)],
        )

        # eef: 每姿态各次回放两两差异(位置 mm / 姿态 deg), 复用 2a 的 eef_pairwise
        pos_list, rot_list = [], []
        if n_runs >= 2:
            for p in range(n_poses):
                poses = [EefPose(np.array(all_runs[r][p]["t_meas"], float),
                                 np.array(all_runs[r][p]["r_meas_rotvec"], float),
                                 np.array(all_runs[r][p]["q_meas"], float))
                         for r in range(n_runs)]
                em = eef_pairwise(poses)
                pos_list.append(em["eef_t_mean_mm"])
                rot_list.append(em["eef_r_mean_deg"])
        pos = np.array(pos_list) if pos_list else np.zeros(1)
        rot = np.array(rot_list) if rot_list else np.zeros(1)
        eef = dict(
            pos_mean_mm=float(pos.mean()), pos_max_mm=float(pos.max()), pos_min_mm=float(pos.min()),
            rot_mean_deg=float(rot.mean()), rot_max_deg=float(rot.max()), rot_min_deg=float(rot.min()),
            per_pose_pos_mean_mm=pos.tolist(),
            per_pose_rot_mean_deg=rot.tolist(),
        )
        return joint, eef

    @staticmethod
    def _plot(joint, eef, n_runs, n_poses, out_png, joint_names):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[plot] matplotlib 不可用, 跳过绘图: {e}")
            return
        short = [_m2._cap._short_joint(n) for n in joint_names]
        x = np.arange(len(short))
        w = 0.27
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

        ax1.bar(x - w, joint["min"], w, label="min", color="#8ecae6")
        ax1.bar(x, joint["mean"], w, label="mean", color="#219ebc")
        ax1.bar(x + w, joint["max"], w, label="max", color="#fb8500")
        for xi, mv in zip(x, joint["mean"]):
            ax1.text(xi, mv, f"{mv:.2f}", ha="center", va="bottom", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(short, rotation=30, ha="right")
        ax1.set_ylabel("joint repeatability (deg)")
        ax1.set_title(f"Per-joint replay repeatability (pairwise across runs)  "
                      f"({n_runs} runs x {n_poses} poses)  "
                      f"overall mean={joint['overall_mean']:.2f}deg  max={joint['overall_max']:.2f}deg")
        ax1.legend()
        ax1.grid(axis="y", ls=":", alpha=0.6)

        labels = ["pos err (mm)", "rot err (deg)"]
        xe = np.arange(len(labels))
        means = [eef["pos_mean_mm"], eef["rot_mean_deg"]]
        maxs = [eef["pos_max_mm"], eef["rot_max_deg"]]
        mins = [eef["pos_min_mm"], eef["rot_min_deg"]]
        ax2.bar(xe - w, mins, w, label="min", color="#8ecae6")
        ax2.bar(xe, means, w, label="mean", color="#219ebc")
        ax2.bar(xe + w, maxs, w, label="max", color="#fb8500")
        for xi, mv in zip(xe, means):
            ax2.text(xi, mv, f"{mv:.2f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(xe)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("EEF repeatability")
        ax2.set_title("End-effector replay repeatability (pairwise across runs)")
        ax2.legend()
        ax2.grid(axis="y", ls=":", alpha=0.6)

        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"[plot] 已保存 {out_png}")

    # ---- 跨次回放重复性对比（复用参考脚本 2a 的算法）-------------------------
    def _stability_analysis(self, all_runs, run_dirs, out_dir):
        """对多次回放做"同样方式"的重复性对比:
        - EEF 两两差异: Δt(mm) / Δr(deg) / Δjoint(deg)
        - 棋盘格角点亚像素位移: findChessboardCorners + cornerSubPix, 跨 replay per-corner shift(px)
        输出 per-frame 表 + 汇总 + 最差 N 帧 corner overlay + 诊断, 并写报告 txt。
        """
        n_runs = len(all_runs)
        if n_runs < 2:
            print("\n[stability] 只有 1 次回放, 跳过跨次对比(至少需要 2 次)。")
            return
        if self.rs is None and not any(
                os.path.isfile(os.path.join(d, all_runs[0][0]["stamp"] + ".png"))
                for d in run_dirs):
            print("\n[stability] 没有回放图(无相机), 仅做 EEF 重复性对比, 跳过角点。")

        pattern = (self.args.pattern_cols, self.args.pattern_rows)
        expected = pattern[0] * pattern[1]
        n_poses = len(all_runs[0])

        report_path = os.path.join(
            out_dir, f"stability_corner_report_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        rf = open(report_path, "w", encoding="utf-8")
        orig_stdout = sys.stdout
        sys.stdout = _Tee(orig_stdout, rf)
        diff_dir = os.path.join(out_dir, "corner_diff_images")
        os.makedirs(diff_dir, exist_ok=True)
        try:
            print("\n" + "=" * 96)
            print(" 跨次回放重复性对比 (EEF 两两差异 + 棋盘格角点亚像素位移) —— 同 2a 方式")
            print("=" * 96)
            print(f" 回放次数={n_runs}  姿态数={n_poses}  "
                  f"棋盘格={pattern[0]}x{pattern[1]} ({expected} 角点)")
            print(f" 角点阈值: ✓<{CORNER_GOOD_PX:.2f}px  ◯<{CORNER_OK_PX:.2f}px  ⚠>{CORNER_WARN_PX:.2f}px")
            hdr = (f"\n  {'stamp':<22s}  {'eef_dt(mm)':>10s}  {'eef_dr°':>8s}  "
                   f"{'eef_dj°':>8s}  {'crn_mean':>9s}  {'crn_max':>9s}  {'crn_p95':>9s}  flag")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

            eef_t, eef_r, eef_j = [], [], []
            crn_mean, crn_max, crn_p95, crn_std = [], [], [], []
            per_frame = []
            n_detect_fail = 0

            for p in range(n_poses):
                stamp = all_runs[0][p]["stamp"]
                poses = [EefPose(np.array(all_runs[r][p]["t_meas"], float),
                                 np.array(all_runs[r][p]["r_meas_rotvec"], float),
                                 np.array(all_runs[r][p]["q_meas"], float))
                         for r in range(n_runs)]
                em = eef_pairwise(poses)
                eef_t.append(em["eef_t_mean_mm"])
                eef_r.append(em["eef_r_mean_deg"])
                if not np.isnan(em["eef_joint_mean_deg"]):
                    eef_j.append(em["eef_joint_mean_deg"])

                corner_sets, base_img = [], None
                for d in run_dirs:
                    ip = Path(os.path.join(d, stamp + ".png"))
                    cs, img = detect_corners(ip, pattern)
                    if cs is not None and cs.shape[0] == expected:
                        corner_sets.append(cs)
                        if base_img is None:
                            base_img = img

                if len(corner_sets) < 2:
                    n_detect_fail += 1
                    print(f"  {stamp:<22s}  {em['eef_t_mean_mm']:>10.4f}  "
                          f"{em['eef_r_mean_deg']:>8.4f}  {em['eef_joint_mean_deg']:>8.4f}  "
                          f"(只有 {len(corner_sets)}/{n_runs} 张检测到角点, 跳过角点)")
                    continue

                st = corner_shift_stats(corner_sets)
                crn_mean.append(st["mean"]); crn_max.append(st["max"])
                crn_p95.append(st["p95"]); crn_std.append(st["std"])
                flag = ("✓" if st["mean"] < CORNER_GOOD_PX else
                        "◯" if st["mean"] < CORNER_OK_PX else
                        "◯ borderline" if st["mean"] < CORNER_WARN_PX else "⚠ WARN")
                print(f"  {stamp:<22s}  {em['eef_t_mean_mm']:>10.4f}  "
                      f"{em['eef_r_mean_deg']:>8.4f}  {em['eef_joint_mean_deg']:>8.4f}  "
                      f"{st['mean']:>9.4f}  {st['max']:>9.4f}  {st['p95']:>9.4f}  {flag}")
                per_frame.append(dict(stamp=stamp, eef_t=em["eef_t_mean_mm"],
                                      eef_r=em["eef_r_mean_deg"], stats=st,
                                      corner_sets=corner_sets, base_img=base_img))

            # 汇总
            print("\n" + "=" * 96)
            print("[汇总]")
            print("=" * 96)
            if eef_t:
                print(f"  EEF 两两差异:")
                print(f"    Δt (mm)  : mean={np.mean(eef_t):.4f}  median={np.median(eef_t):.4f}  "
                      f"max={np.max(eef_t):.4f}  std={np.std(eef_t):.4f}")
                print(f"    Δr (°)   : mean={np.mean(eef_r):.4f}  median={np.median(eef_r):.4f}  "
                      f"max={np.max(eef_r):.4f}  std={np.std(eef_r):.4f}")
                if eef_j:
                    print(f"    Δjoint(°): mean={np.mean(eef_j):.4f}  max={np.max(eef_j):.4f}")
            if crn_mean:
                print(f"\n  角点亚像素位移 (跨 replay per-corner shift, px):")
                print(f"    mean: mean={np.mean(crn_mean):.4f}  median={np.median(crn_mean):.4f}  max={np.max(crn_mean):.4f}")
                print(f"    max : mean={np.mean(crn_max):.4f}  max={np.max(crn_max):.4f}")
                print(f"    p95 : mean={np.mean(crn_p95):.4f}  max={np.max(crn_p95):.4f}")
                total = len(crn_mean)
                n_good = sum(1 for v in crn_mean if v < CORNER_GOOD_PX)
                n_ok = sum(1 for v in crn_mean if v < CORNER_OK_PX)
                n_warn = sum(1 for v in crn_mean if v > CORNER_WARN_PX)
                print(f"\n  达标分布(按 mean shift): ✓{n_good}/{total}  ◯{n_ok}/{total}  ⚠{n_warn}/{total}")
            if n_detect_fail:
                print(f"\n  注: {n_detect_fail} 帧角点检测不足(检查 --pattern-cols/--pattern-rows)。")

            # 最差 N 帧 overlay
            if per_frame:
                per_frame.sort(key=lambda r: -r["stats"]["mean"])
                print("\n" + "=" * 96)
                print(f"[最差 {self.args.n_worst} 帧 (按角点 mean shift)]  -> corner_diff_images/")
                print("=" * 96)
                for rank, row in enumerate(per_frame[:self.args.n_worst]):
                    st = row["stats"]
                    ovl = Path(os.path.join(diff_dir, row["stamp"] + "__corners_overlay.png"))
                    save_corner_overlay(row["base_img"], row["corner_sets"], ovl)
                    print(f"  rank{rank+1}: {row['stamp']}  corner mean={st['mean']:.4f}px "
                          f"max={st['max']:.4f}px  eef Δt={row['eef_t']:.3f}mm  -> {ovl.name}")

            # 诊断
            print("\n" + "=" * 96)
            print("[诊断]")
            print("=" * 96)
            eef_overall = float(np.mean(eef_t)) if eef_t else float("nan")
            crn_overall = float(np.mean(crn_mean)) if crn_mean else float("nan")
            if eef_t:
                if eef_overall < 1.0:
                    print(f"  ✓ EEF 重复性好 (Δt mean={eef_overall:.3f}mm < 1mm)")
                elif eef_overall < 3.0:
                    print(f"  ◯ EEF 重复性一般 (Δt mean={eef_overall:.3f}mm, 1-3mm)")
                else:
                    print(f"  ✗ EEF 重复性差 (Δt mean={eef_overall:.3f}mm > 3mm)")
            if crn_mean:
                if crn_overall < CORNER_OK_PX:
                    print(f"  ✓ 角点稳 (mean={crn_overall:.3f}px), 相机/板/臂在图像空间重复性好")
                elif crn_overall < CORNER_WARN_PX:
                    print(f"  ◯ 角点 borderline (mean={crn_overall:.3f}px)")
                else:
                    print(f"  ⚠ 角点不稳 (mean={crn_overall:.3f}px): 查相机支架/对焦/曝光/光源闪烁")
                if eef_t and eef_overall < 1.0 and crn_overall >= CORNER_WARN_PX:
                    print("  ⇒ 机械臂稳但角点不稳 → 元凶在相机端, 不是机械臂。")
            print(f"\n[stability] 报告: {report_path}")
        finally:
            sys.stdout = orig_stdout
            rf.close()

    # ---- 主流程 --------------------------------------------------------------
    def run(self):
        self.init_dds()

        # 先加载记录(顺便确定被回放臂 self.arm / self.joint_names), 再按臂初始化控制/建模
        records = self._load_records_full()
        if not records:
            print(f"ERROR: 在 {self.args.data}/txt 没找到可回放的记录 txt。")
            return
        self.init_targets()
        self.init_kinematics()
        print(f"找到 {len(records)} 个姿态, 将回放 {self.args.runs} 次。(臂: {self.arm})")

        replay_root = os.path.join(self.args.data, "replay")
        os.makedirs(replay_root, exist_ok=True)

        # 相机可选：连不上也能跑(只统计关节/eef, 不存图)
        self.rs = ZmqCamera(self.args.host, self.args.port, self.args.intrinsics)
        if self.rs.wait_for_stream(self.args.cam_timeout):
            print("相机流正常, 回放时会存图。")
        else:
            self.rs.stop()
            self.rs = None
            print("WARNING: 未收到相机流, 将只统计关节/eef 误差, 不存回放图。")

        # 相机探活后: 把腰部 rpy 设为 0,0,0 (接管后随 weight 渐入回正)
        self._zero_waist()

        arm_cn = "左臂" if self.arm == "left" else "右臂"
        try:
            input(f"\n将以 {self.args.speed} rad/s 慢速回放 {len(records)} 个姿态 x "
                  f"{self.args.runs} 次。请清空{arm_cn}工作区、备好急停(Ctrl-C), 按 Enter 开始...")
        except KeyboardInterrupt:
            if self.rs:
                self.rs.stop()
            print("\n已取消, 未接管手臂。")
            return

        self.ctrl = RecurrentThread(interval=CONTROL_DT, target=self.control_step, name="arm_replay")
        self.ctrl.Start()
        print(f"Engaging arm_sdk (weight 0->1, {T_ENGAGE}s) ...\n")
        time.sleep(T_ENGAGE + 0.3)

        all_runs = []
        run_dirs = []
        try:
            for k in range(self.args.runs):
                run_ts = time.strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join(replay_root, run_ts)
                print(f"\n===== 第 {k+1}/{self.args.runs} 次回放  ->  replay/{run_ts} =====")
                run_dirs.append(run_dir)
                all_runs.append(self._run_once(records, run_dir))
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n收到中断, 停止回放。")
        finally:
            try:
                self.release()
            except Exception as e:
                print(f"\n[release] 交还时异常(忽略): {e}")
            if self.rs:
                self.rs.stop()

        if not all_runs:
            print("没有完成任何回放, 不做统计。")
            return

        # 统一各次回放长度（若中途中断, 取最短）
        n = min(len(run) for run in all_runs)
        all_runs = [run[:n] for run in all_runs]
        run_dirs = run_dirs[:len(all_runs)]
        joint, eef = self._aggregate(all_runs)

        summary = dict(
            data_dir=os.path.abspath(self.args.data),
            arm=self.arm,
            n_runs=len(all_runs), n_poses=n,
            ee_frame=self.args.ee_frame,
            joint_names=self.joint_names,
            joint_repeatability_deg=joint,
            eef_repeatability=eef,
        )
        with open(os.path.join(replay_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n========== 统计结果(多次回放之间的重复性) ==========")
        print(f"回放次数={len(all_runs)}  姿态数={n}")
        print(f"[joint] 整体 mean={joint['overall_mean']:.3f}deg  max={joint['overall_max']:.3f}deg")
        print(f"[joint] 每姿态 mean(deg): "
              + ", ".join(f"{v:.3f}" for v in joint["per_pose_mean"]))
        print(f"[eef ] pos mean={eef['pos_mean_mm']:.2f}mm  max={eef['pos_max_mm']:.2f}mm")
        print(f"[eef ] rot mean={eef['rot_mean_deg']:.3f}deg  max={eef['rot_max_deg']:.3f}deg")
        print("每关节重复性(度, 回放间两两差异): name  mean / max / min")
        for nm, me, mx, mn in zip(self.joint_names, joint["mean"], joint["max"], joint["min"]):
            print(f"  {nm:26s} {me:6.3f} / {mx:6.3f} / {mn:6.3f}")

        self._plot(joint, eef, len(all_runs), n,
                   os.path.join(replay_root, "joint_error_stats.png"), self.joint_names)

        # 跨次回放重复性对比（参考 2a：EEF 两两差异 + 棋盘格角点亚像素位移）
        self._stability_analysis(all_runs, run_dirs, replay_root)

        print(f"\nDone. 结果在 {replay_root}")


if __name__ == "__main__":
    _DEFAULT_URDF = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "g1_description", "g1_29dof_with_hand_rev_1_0.urdf"))

    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="采集数据目录(含 <时间戳>.png + txt/<时间戳>.txt)，如 ./calib_data_20260615_0416")
    ap.add_argument("--runs", type=int, default=3, help="回放次数 (默认 3)")
    ap.add_argument("--net", default="enp12s0", help="上位机连机器人的网卡, 如 enp12s0")
    ap.add_argument("--arm", choices=["left", "right"], default=None,
                    help="回放哪条臂; 默认按数据 txt 里的 arm 字段自动识别")
    ap.add_argument("--speed", type=float, default=_m2.DEFAULT_SPEED,
                    help=f"左臂每关节最大角速度 rad/s (越小越慢越安全, 默认 {_m2.DEFAULT_SPEED})")
    ap.add_argument("--settle", type=float, default=_m2.DEFAULT_SETTLE,
                    help=f"到位后等实测角稳定的最长秒数 (默认 {_m2.DEFAULT_SETTLE})")
    ap.add_argument("--tol-deg", type=float, default=1.5, help="实测关节角到位阈值(度) (默认 1.5)")
    ap.add_argument("--ee-frame", default=None,
                    help="末端坐标系(做 FK eef 比较); 不填则按臂取 <arm>_wrist_yaw_link")
    ap.add_argument("--urdf", default=_DEFAULT_URDF, help="G1 29-DoF URDF (FK/重力补偿用)")
    ap.add_argument("--gravity-scale", type=float, default=1.0, help="重力补偿力矩缩放 (默认 1.0)")
    ap.add_argument("--host", default="192.168.123.164", help="Jetson(image_server) IP")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--intrinsics", default=None, help="相机内参 json (可选)")
    ap.add_argument("--cam-timeout", type=float, default=8.0, help="等待相机图像流的超时(s)")
    ap.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_COLS,
                    help=f"棋盘格横向内角点数 (默认 {DEFAULT_PATTERN_COLS}; OpenCV 指内角点不是格子数)")
    ap.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_ROWS,
                    help=f"棋盘格纵向内角点数 (默认 {DEFAULT_PATTERN_ROWS})")
    ap.add_argument("--n-worst", type=int, default=DEFAULT_N_WORST,
                    help=f"保存最差 N 帧的角点叠加图 (默认 {DEFAULT_N_WORST})")
    ap.set_defaults(gravcomp=True)
    args = ap.parse_args()

    _arm_hint = args.arm or "数据自动识别"
    print(f"WARNING: clear the workspace; keep a hand ready to support the arm ({_arm_hint}).")
    print(f"将慢速回放 {os.path.abspath(args.data)} 里的姿态 {args.runs} 次 (臂: {_arm_hint})。")
    input("Robot must be standing (锁定站立), motion mode NOT released. Press Enter...")
    ChannelFactoryInitialize(0, args.net)
    ReplayMultiRun(args).run()
