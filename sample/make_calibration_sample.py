"""生成跨夜通量校正用的虚构样例。

三个文件:
  sample/reference_bridge.csv  稳定参照星测量(night_id, star_id, flux),
                               可跨夜重复,含「跨夜桥接」——同一颗星在
                               相邻夜晚各出现一次,把各夜串成一张连通图;
  sample/reference_conflict.csv 与 bridge 相同的构图,但故意让 S_ANCHOR
                               在第 3 夜偏亮 12%、S_BRIDGE 在第 5 夜偏暗
                               12%,相互矛盾,用来演示冲突夜晚/星号的
                               报错(配合默认最大相对偏差 5%);
  sample/target_star.csv       目标星测量(night_id, time_day, flux),
                               含一个参照图中没有、因而未校准的第 7 夜。

虚构设定(全部为假数据):
  参照星固有通量  S_ANCHOR=100, S_BRIDGE=60, S_C=40
  六夜透过率      N1=0.90, N2=0.80, N3=1.10, N4=0.95, N5=1.05, N6=0.85
  目标星真实通量  8 + 0.4*sin(2πt/3.7),必须先被当夜透过率「污染」,
                  校正后才能还原到同一尺度。
参照测量取整到 4 位小数,不叠加随机噪声,保证清洁样例在很小的偏差
限下也能通过、矛盾样例的偏差只来自故意注入的两处。
"""
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(20261114)

INTRINSIC = {"S_ANCHOR": 100.0, "S_BRIDGE": 60.0, "S_C": 40.0}
THROUGHPUT = {"N1": 0.90, "N2": 0.80, "N3": 1.10, "N4": 0.95,
              "N5": 1.05, "N6": 0.85, "N7": 1.20}

# (night, star):清洁样例里每夜都有锚星测量,另有两颗参照星反复出现,
# 形成「同一颗星跨夜重复」与「同一夜多颗星交叉」的桥接结构。
PLAN = [
    ("N1", "S_ANCHOR"), ("N1", "S_BRIDGE"), ("N1", "S_C"),
    ("N2", "S_ANCHOR"), ("N2", "S_BRIDGE"), ("N2", "S_C"),
    ("N3", "S_ANCHOR"), ("N3", "S_BRIDGE"), ("N3", "S_C"),
    ("N4", "S_ANCHOR"), ("N4", "S_BRIDGE"), ("N4", "S_C"),
    ("N5", "S_ANCHOR"), ("N5", "S_BRIDGE"), ("N5", "S_C"),
    ("N6", "S_ANCHOR"), ("N6", "S_BRIDGE"), ("N6", "S_C"),
]

# 矛盾样例相对清洁样例的乘性偏差(仅这两处)
CONFLICT_FACTORS = {("N3", "S_ANCHOR"): 1.12, ("N5", "S_BRIDGE"): 0.88}


def observed(night: str, star: str) -> float:
    """理想观测通量 = 固有通量 × 当夜透过率(取 4 位小数)。"""
    return round(INTRINSIC[star] * THROUGHPUT[night], 4)


def write_reference(path: Path, rows: list[tuple[str, str, float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("night_id,star_id,flux\n")
        for night, star, flux in rows:
            fh.write(f"{night},{star},{flux:.4f}\n")
    print(f"已写出 {path}({len(rows)} 行)")


def main() -> None:
    # 1) 清洁样例:跨夜桥接,六夜全部与锚星连通
    clean = [(n, s, observed(n, s)) for n, s in PLAN]
    write_reference(DIR / "reference_bridge.csv", clean)

    # 2) 矛盾样例:同一份构图,两处刻意偏离 → 不同夜晚/星号互相矛盾
    conflict = [
        (n, s, round(f * CONFLICT_FACTORS.get((n, s), 1.0), 4))
        for n, s, f in clean
    ]
    write_reference(DIR / "reference_conflict.csv", conflict)

    # 3) 目标星:N1..N6 已校准,N7 只有目标星 → 未校准
    rows = ["night_id,time_day,flux"]
    n_per_night = {"N1": 22, "N2": 20, "N3": 18, "N4": 21,
                   "N5": 19, "N6": 20, "N7": 14}
    t0 = 0.0
    for night, count in n_per_night.items():
        # 每夜约占 0.9 天的一段连续观测窗,夜与夜之间相隔约 1 天
        times = np.sort(rng.uniform(t0, t0 + 0.9, size=count))
        for t in times:
            intrinsic = 8.0 + 0.4 * np.sin(2 * np.pi * t / 3.7)
            flux = intrinsic * THROUGHPUT[night] * (1.0 + rng.normal(0.0, 0.008))
            rows.append(f"{night},{t:.6f},{flux:.6f}")
        t0 += 1.0
    with (DIR / "target_star.csv").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"已写出 {DIR / 'target_star.csv'}"
          f"({len(rows) - 1} 行,共 7 夜,第 7 夜未校准)")


if __name__ == "__main__":
    main()
