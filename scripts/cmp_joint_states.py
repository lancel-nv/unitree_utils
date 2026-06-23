"""比较 joint_states_record/ 里多份关节状态记录的一致性。

读取若干份由 get_joint_states.py 保存的 JSON 记录，按关节逐个统计
各次记录之间的差异（默认看角度 q），输出每个关节的均值/极差/标准差，
并按极差从大到小排序，方便快速看出哪些关节在多次记录间最不一致。

用法:
    # 比较 joint_states_record/ 下全部记录
    python cmp_joint_states.py

    # 只比较指定的几份记录
    python cmp_joint_states.py a.json b.json c.json

    # 指定比较的字段 (q / dq / tau_est, 默认 q)
    python cmp_joint_states.py --field q

    # 自定义记录目录
    python cmp_joint_states.py --dir /path/to/joint_states_record
"""

import argparse
import glob
import json
import math
import os

# 默认的记录目录（位于项目根目录下，与 get_joint_states.py 保存路径一致）
DEFAULT_RECORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "joint_states_record",
)


def load_record(path):
    """读取单份 JSON 记录，返回 (timestamp, {joint_name: {q, dq, tau_est}})。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    joints = {}
    for j in data.get("joints", []):
        joints[j["name"]] = {
            "q": j.get("q"),
            "dq": j.get("dq"),
            "tau_est": j.get("tau_est"),
        }
    ts = data.get("timestamp", os.path.basename(path))
    return ts, joints


def mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser(description="比较多份关节状态记录的一致性")
    parser.add_argument("files", nargs="*", help="要比较的 JSON 文件 (不指定则比较目录下全部)")
    parser.add_argument("--dir", default=DEFAULT_RECORD_DIR, help="记录目录")
    parser.add_argument("--field", default="q", choices=["q", "dq", "tau_est"],
                        help="比较的字段, 默认 q (角度)")
    args = parser.parse_args()

    # 收集要比较的文件
    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.json")))

    if len(files) < 2:
        print(f"至少需要 2 份记录才能比较，当前只找到 {len(files)} 份: {files}")
        return

    field = args.field

    # 加载所有记录
    records = []  # [(ts, joints_dict), ...]
    for f in files:
        try:
            records.append(load_record(f))
        except Exception as e:
            print(f"跳过无法读取的文件 {f}: {e}")

    if len(records) < 2:
        print("有效记录不足 2 份，无法比较。")
        return

    print("=" * 90)
    print(f"比较字段: {field}    记录份数: {len(records)}")
    print("参与比较的记录:")
    for ts, _ in records:
        print(f"  - {ts}")
    print("=" * 90)

    # 以第一份记录的关节名为基准（各份关节名应一致）
    base_names = list(records[0][1].keys())

    # 逐关节收集各记录的值
    rows = []  # (name, mean, std, vmin, vmax, span, values)
    for name in base_names:
        values = []
        for _, joints in records:
            jd = joints.get(name)
            if jd is None or jd.get(field) is None:
                values.append(None)
            else:
                values.append(float(jd[field]))

        valid = [v for v in values if v is not None]
        if not valid:
            continue
        mean, std = mean_std(valid)
        vmin, vmax = min(valid), max(valid)
        span = vmax - vmin
        rows.append((name, mean, std, vmin, vmax, span, values))

    # 按极差从大到小排序：差异最大的关节排在最前面
    rows.sort(key=lambda r: r[5], reverse=True)

    unit = "rad" if field == "q" else ("rad/s" if field == "dq" else "Nm")
    print(f"\n各关节 {field} 在多次记录间的一致性 (按极差降序，单位 {unit}):\n")
    header = f"{'joint':<20} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'span(max-min)':>14}"
    print(header)
    print("-" * len(header))
    for name, mean, std, vmin, vmax, span, _ in rows:
        print(f"{name:<20} {mean:>10.4f} {std:>10.4f} {vmin:>10.4f} {vmax:>10.4f} {span:>14.4f}")

    # 汇总
    spans = [r[5] for r in rows]
    if spans:
        max_row = rows[0]
        avg_span = sum(spans) / len(spans)
        print("\n" + "=" * 90)
        print("一致性汇总:")
        print(f"  最大极差关节: {max_row[0]}  span={max_row[5]:.4f} {unit}")
        print(f"  平均极差    : {avg_span:.4f} {unit}")
        print(f"  最大标准差  : {max(r[2] for r in rows):.4f} {unit}")
        print("=" * 90)


if __name__ == "__main__":
    main()
