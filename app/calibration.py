"""跨夜通量校正:联合估计各夜透过率。

模型
----
同一颗参照星在不同夜晚的观测通量满足

    观测通量 = 固有通量 × 当夜透过率

取对数后是加性模型 ``ln f_ij = ln F_i + ln T_j``(i 为星,j 为夜)。
把参照测量看成「星—夜」二部图的边,从锚星出发做连通性搜索:

* 只有与锚星连通的夜晚才参与校正,记为可达夜晚;
* 以用户给出的锚星固有通量为基准(把 ``ln F_锚星`` 固定,消除整体
  尺度自由度),在可达连通分量内联合求解各星固有通量与各夜透过率;
* 求解采用按相对残差反比的迭代重加权最小二乘:最小二乘会把矛盾摊薄
  到整张图,重加权让真正互相矛盾的测量在残差里自己显形;逐条检查
  ``|观测值 - 预测值| / 预测值``,超过最大相对偏差即判为冲突,列出
  相应的夜晚与星号,不生成校正文件;
* 与锚星不连通的夜晚列为未校准,其目标观测不进入结果。

本模块不依赖 scipy:加权正规方程是稀疏、对称正定的,用预条件共轭
梯度法求解。
"""
from __future__ import annotations

import io
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

MAX_REF_ROWS = 50_000
MAX_TARGET_ROWS = 50_000
MIN_ROWS = 10          # 与 app.main.MIN_ROWS 一致:原周期入口的最少行数
_DEV_TOL = 1e-12       # 判定偏差是否超限时允许的浮点余量
_MAX_IRLS = 100       # 稳健重加权最多迭代轮数
_IRLS_FLOOR = 1e-3    # 稳健权重下限,避免小残差测量独占权重


class CalibrationError(Exception):
    """带 HTTP 状态码与结构化说明的校正错误。"""

    def __init__(self, status_code: int, code: str, message: str,
                 extra: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail: dict = {"code": code, "message": message}
        if extra:
            self.detail.update(extra)


# --------------------------------------------------------------------------- #
# 解析与校验
# --------------------------------------------------------------------------- #

def _natural_key(text: str) -> tuple:
    """让 N2 排在 N10 之前;同时适用于纯数字或任意字符串。"""
    parts = re.split(r"(\d+)", str(text))
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower())
                 for p in parts)


def _read_csv(raw: bytes, required: tuple[str, ...], label: str) -> pd.DataFrame:
    try:
        # utf-8-sig:兼容 Excel 导出的带 BOM 文件
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                         encoding="utf-8-sig")
    except Exception as exc:  # 空文件、编码错误等
        raise CalibrationError(400, "csv_parse", f"{label} CSV 解析失败:{exc}")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise CalibrationError(
            400, "csv_missing_columns",
            f"{label} CSV 缺少必需列:{', '.join(missing)}"
            f"(需要 {', '.join(required)})")
    if len(df) == 0:
        raise CalibrationError(400, "csv_empty", f"{label} CSV 没有数据行")
    return df


def _row_numbers(index: pd.Index, limit: int = 8) -> str:
    """把 0 基行号换成 CSV 文件行号(表头算第 1 行)便于定位。"""
    rows = [int(i) + 2 for i in index[:limit]]
    text = "、".join(str(r) for r in rows)
    if len(index) > limit:
        text += " 等"
    return text


def _id_column(df: pd.DataFrame, col: str, label: str) -> np.ndarray:
    values = df[col].astype(str).str.strip().to_numpy(dtype=object)
    bad = np.array([v == "" for v in values])
    if bad.any():
        raise CalibrationError(
            400, f"{label}_id_empty",
            f"{label} CSV 的 {col} 存在空值(第 {_row_numbers(df.index[bad])} 行)")
    return values


def _numeric_column(df: pd.DataFrame, col: str, label: str,
                    positive: bool) -> np.ndarray:
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        raise CalibrationError(
            400, f"{label}_{col}_invalid",
            f"{label} CSV 的 {col} 必须全部为有限数值"
            f"(第 {_row_numbers(df.index[bad])} 行存在空值/非数值/NaN/Inf)")
    if positive:
        bad = values <= 0.0
        if bad.any():
            raise CalibrationError(
                400, f"{label}_{col}_nonpositive",
                f"{label} CSV 的 {col} 必须为正数"
                f"(第 {_row_numbers(df.index[bad])} 行不为正)")
    return values


def parse_reference(raw: bytes) -> dict:
    """解析参照星测量 CSV:night_id、star_id、flux。"""
    df = _read_csv(raw, ("night_id", "star_id", "flux"), "参照星测量")
    if len(df) > MAX_REF_ROWS:
        raise CalibrationError(400, "reference_too_many_rows",
                               f"参照测量行数超过上限 {MAX_REF_ROWS}")
    night = _id_column(df, "night_id", "参照测量")
    star = _id_column(df, "star_id", "参照测量")
    flux = _numeric_column(df, "flux", "参照测量", positive=True)

    dup = pd.DataFrame({"night_id": night, "star_id": star}).duplicated(
        keep=False).to_numpy()
    if dup.any():
        pairs = list(dict.fromkeys(
            (night[i], star[i]) for i in np.flatnonzero(dup)))
        shown = "、".join(f"({n}, {s})" for n, s in pairs[:8])
        if len(pairs) > 8:
            shown += " 等"
        raise CalibrationError(
            400, "reference_duplicate_measurement",
            f"同一夜晚对同一参照星出现多条测量,无法确定使用哪一条:{shown}",
            {"duplicates": [{"night_id": n, "star_id": s} for n, s in pairs]})
    return {"night": night, "star": star, "flux": flux}


def parse_target(raw: bytes) -> dict:
    """解析目标星测量 CSV:night_id、time_day、flux。"""
    df = _read_csv(raw, ("night_id", "time_day", "flux"), "目标星测量")
    if len(df) > MAX_TARGET_ROWS:
        raise CalibrationError(400, "target_too_many_rows",
                               f"目标测量行数超过上限 {MAX_TARGET_ROWS}")
    night = _id_column(df, "night_id", "目标测量")
    time_day = _numeric_column(df, "time_day", "目标测量", positive=False)
    flux = _numeric_column(df, "flux", "目标测量", positive=True)
    return {"night": night, "time_day": time_day, "flux": flux}


# --------------------------------------------------------------------------- #
# 连通性与联合估计
# --------------------------------------------------------------------------- #

def _reachable(ref: dict, anchor: str) -> tuple[set[str], set[str]]:
    """从锚星出发,返回 (可达星集合, 可达夜晚集合)。"""
    star_to_nights: dict[str, set[str]] = defaultdict(set)
    night_to_stars: dict[str, set[str]] = defaultdict(set)
    for night, star in zip(ref["night"], ref["star"]):
        star_to_nights[star].add(night)
        night_to_stars[night].add(star)

    seen_stars = {anchor}
    seen_nights: set[str] = set()
    stack = [anchor]
    while stack:
        star = stack.pop()
        for night in star_to_nights.get(star, ()):
            if night in seen_nights:
                continue
            seen_nights.add(night)
            for other in night_to_stars[night]:
                if other not in seen_stars:
                    seen_stars.add(other)
                    stack.append(other)
    return seen_stars, seen_nights


def _pcg(matvec, rhs: np.ndarray, diag: np.ndarray,
         x0: np.ndarray | None = None,
         tol: float = 1e-13, maxiter: int | None = None) -> np.ndarray:
    """预条件共轭梯度法,求解对称正定系统 M x = rhs。"""
    n = rhs.size
    x = np.zeros(n, dtype=float) if x0 is None else x0.astype(float).copy()
    norm_rhs = float(np.linalg.norm(rhs))
    if norm_rhs == 0.0:
        return np.zeros(n, dtype=float)
    r = rhs - matvec(x)
    z = r / diag
    p = z.copy()
    rz = float(r @ z)
    limit = maxiter if maxiter is not None else 2 * n + 100
    for _ in range(limit):
        ap = matvec(p)
        pap = float(p @ ap)
        if not math.isfinite(pap) or pap <= 0.0:
            raise CalibrationError(500, "fit_failed", "透过率联合估计数值失败")
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        if float(np.linalg.norm(r)) <= tol * norm_rhs:
            return x
        z = r / diag
        rz_new = float(r @ z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    # 数值上未完全收敛时,只要残差足够小仍可采用
    if float(np.linalg.norm(r)) <= 1e-8 * norm_rhs:
        return x
    raise CalibrationError(500, "fit_failed",
                           "透过率联合估计未收敛,请检查参照测量")


def fit_throughputs(ref: dict, anchor: str, anchor_flux: float,
                    max_rel_dev: float) -> dict:
    """在锚星可达分量内联合估计透过率,并检查参照测量偏差。"""
    stars_all = set(ref["star"].tolist())
    if anchor not in stars_all:
        known = sorted(stars_all, key=_natural_key)
        raise CalibrationError(
            400, "anchor_missing",
            f"锚星 {anchor} 未出现在参照测量中",
            {"anchor_star_id": anchor, "known_stars": known[:20],
             "n_known_stars": len(known)})

    seen_stars, seen_nights = _reachable(ref, anchor)
    star = ref["star"]
    night = ref["night"]
    flux = ref["flux"]
    log_flux = np.log(flux)

    used = np.array([s in seen_stars for s in star], dtype=bool)
    if not used.any():  # 锚星存在则其自身测量必然属于该分量
        raise CalibrationError(500, "fit_failed", "可达参照测量为空")

    free_stars = sorted((s for s in seen_stars if s != anchor), key=_natural_key)
    nights = sorted(seen_nights, key=_natural_key)
    s_index = {s: i for i, s in enumerate(free_stars)}
    n_index = {n: i for i, n in enumerate(nights)}
    n_star, n_night = len(free_stars), len(nights)

    used_night = night[used]
    used_star = star[used]
    used_log = log_flux[used]
    night_idx = np.array([n_index[n] for n in used_night], dtype=np.int64)
    is_anchor = used_star == anchor
    free_row = ~is_anchor
    free_star_idx = np.array([s_index[s] for s in used_star[free_row]],
                             dtype=np.int64)
    free_night_idx = night_idx[free_row]
    free_log = used_log[free_row]

    # 正规方程:M [ln F_free, ln T] = rhs
    k_star = np.bincount(free_star_idx, minlength=n_star).astype(float)
    m_night = np.bincount(night_idx, minlength=n_night).astype(float)
    rhs_star = np.bincount(free_star_idx, weights=free_log,
                           minlength=n_star)
    anchor_A = math.log(anchor_flux)
    rhs_night = (np.bincount(night_idx, weights=used_log, minlength=n_night)
                 - np.bincount(night_idx[is_anchor], minlength=n_night) * anchor_A)
    rhs = np.concatenate([rhs_star, rhs_night])
    diag = np.concatenate([k_star, m_night])
    if np.any(diag <= 0):
        raise CalibrationError(500, "fit_failed", "参照测量图结构异常")

    def matvec(x: np.ndarray) -> np.ndarray:
        a = x[:n_star]
        b = x[n_star:]
        out_a = np.bincount(free_star_idx, weights=b[free_night_idx],
                            minlength=n_star) + k_star * a
        out_b = np.bincount(free_night_idx, weights=a[free_star_idx],
                            minlength=n_night) + m_night * b
        return np.concatenate([out_a, out_b])

    # 先用一次普通最小二乘给出初值,再做加权迭代:矛盾的参照测量会把
    # 整体尺度带偏,用残差反比加权逐轮压低离群测量的影响,使真正互相
    # 矛盾的「夜晚/星号」在残差里自己显形。
    x = _pcg(matvec, rhs, diag)
    n_used = int(used.sum())
    weights = np.ones(n_used, dtype=float)
    rel_dev = np.zeros(n_used, dtype=float)
    for _ in range(_MAX_IRLS):
        star_A = np.empty(n_used, dtype=float)
        star_A[free_row] = x[:n_star][free_star_idx]
        star_A[is_anchor] = anchor_A
        predicted = np.exp(star_A + x[n_star:][night_idx])
        rel_dev = np.abs(flux[used] - predicted) / predicted
        # 尺度无关的稳健权重:相对残差的倒数,加下限避免权重爆炸
        new_weights = 1.0 / np.maximum(rel_dev, _IRLS_FLOOR)
        if np.allclose(new_weights, weights, rtol=1e-6, atol=1e-9):
            break
        weights = new_weights
        w_free = weights[free_row]
        k_star = np.bincount(free_star_idx, weights=w_free,
                             minlength=n_star)
        m_night = np.bincount(night_idx, weights=weights, minlength=n_night)
        rhs_star = np.bincount(free_star_idx, weights=w_free * free_log,
                               minlength=n_star)
        rhs_night = (np.bincount(night_idx, weights=weights * used_log,
                                 minlength=n_night)
                     - np.bincount(night_idx[is_anchor],
                                   weights=weights[is_anchor],
                                   minlength=n_night) * anchor_A)
        rhs = np.concatenate([rhs_star, rhs_night])
        diag = np.concatenate([k_star, m_night])
        if np.any(diag <= 0):
            raise CalibrationError(500, "fit_failed", "参照测量图结构异常")

        def matvec_w(xw: np.ndarray) -> np.ndarray:
            a = xw[:n_star]
            b = xw[n_star:]
            out_a = np.bincount(free_star_idx, weights=w_free * b[free_night_idx],
                                minlength=n_star) + k_star * a
            out_b = (np.bincount(free_night_idx, weights=weights[free_row] * a[free_star_idx],
                                 minlength=n_night) + m_night * b)
            return np.concatenate([out_a, out_b])

        x = _pcg(matvec_w, rhs, diag, x0=x)
    free_A = x[:n_star]
    log_through = x[n_star:]

    # 用稳健拟合的结果重算逐条偏差;预测值非有限(下溢/上溢)一律记为冲突
    star_A = np.empty(n_used, dtype=float)
    star_A[free_row] = free_A[free_star_idx]
    star_A[is_anchor] = anchor_A
    predicted = np.exp(star_A + log_through[night_idx])
    safe_pred = np.isfinite(predicted) & (predicted > 0.0)
    rel_dev = np.full(n_used, np.inf, dtype=float)
    rel_dev[safe_pred] = (np.abs(flux[used][safe_pred] - predicted[safe_pred])
                          / predicted[safe_pred])

    conflicts = []
    for i in np.flatnonzero(rel_dev > max_rel_dev + _DEV_TOL):
        conflicts.append({
            "night_id": str(used_night[i]),
            "star_id": str(used_star[i]),
            "observed_flux": float(flux[used][i]),
            "predicted_flux": float(predicted[i]),
            "rel_dev": float(rel_dev[i]),
        })
    conflicts.sort(key=lambda c: (_natural_key(c["night_id"]),
                                  _natural_key(c["star_id"])))

    star_fluxes = {anchor: float(anchor_flux)}
    star_fluxes.update({s: float(math.exp(free_A[s_index[s]]))
                        for s in free_stars})
    throughputs = {n: float(math.exp(log_through[n_index[n]]))
                   for n in nights}

    used_night_counts = Counter(str(n) for n in used_night)
    return {
        "used_stars": seen_stars,
        "used_nights": nights,
        "throughputs": throughputs,
        "star_fluxes": star_fluxes,
        "n_used": int(used.sum()),
        "n_excluded": int((~used).sum()),
        "used_night_counts": used_night_counts,
        "conflicts": conflicts,
        "max_rel_dev": float(max_rel_dev),
    }


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def run_calibration(reference_raw: bytes, target_raw: bytes,
                    anchor_star_id: str, anchor_flux: float,
                    max_rel_dev: float) -> dict:
    """完成一次校正,成功返回报告与 CSV 文本;失败抛 CalibrationError。"""
    anchor = str(anchor_star_id).strip()
    if not anchor:
        raise CalibrationError(400, "anchor_invalid", "锚星星号不能为空")
    if not math.isfinite(anchor_flux) or anchor_flux <= 0.0:
        raise CalibrationError(400, "anchor_flux_invalid",
                               "锚星固有通量必须为正的有限数值")
    if not math.isfinite(max_rel_dev) or max_rel_dev < 0.0:
        raise CalibrationError(400, "max_rel_dev_invalid",
                               "最大相对偏差必须为非负的有限数值")

    ref = parse_reference(reference_raw)
    tgt = parse_target(target_raw)
    fit = fit_throughputs(ref, anchor, anchor_flux, max_rel_dev)

    if fit["conflicts"]:
        nights = sorted({c["night_id"] for c in fit["conflicts"]},
                        key=_natural_key)
        stars = sorted({c["star_id"] for c in fit["conflicts"]},
                       key=_natural_key)
        # 即便本次失败,也把与锚星不连通、不会进入结果的夜晚一并交代清楚
        ref_nights = set(ref["night"].tolist())
        calibrated = set(fit["used_nights"])
        drop = Counter(str(n) for n in tgt["night"] if n not in calibrated)
        uncalibrated = [
            {"night_id": night,
             "n_target_rows": int(drop[night]),
             "reason": ("disconnected_from_anchor" if night in ref_nights
                        else "not_in_reference")}
            for night in sorted(drop, key=_natural_key)
        ]
        raise CalibrationError(
            400, "reference_conflict",
            "参照测量无法同时满足最大相对偏差 "
            f"{max_rel_dev:g};冲突夜晚:{', '.join(nights)};"
            f"冲突星号:{', '.join(stars)}。未生成校正文件。",
            {
                "conflicts": fit["conflicts"],
                "conflict_nights": nights,
                "conflict_stars": stars,
                "max_rel_dev": float(max_rel_dev),
                "uncalibrated_nights": uncalibrated,
            })

    throughputs = fit["throughputs"]
    calibrated_night_set = set(throughputs)
    ref_night_set = set(ref["night"].tolist())

    keep = np.array([n in calibrated_night_set for n in tgt["night"]],
                    dtype=bool)
    cal_night = tgt["night"][keep]
    cal_time = tgt["time_day"][keep]
    cal_flux = tgt["flux"][keep]
    if cal_time.size:
        corr_flux = cal_flux / np.array(
            [throughputs[n] for n in cal_night], dtype=float)
    else:
        corr_flux = np.empty(0, dtype=float)

    order = np.argsort(cal_time, kind="stable")
    cal_night = cal_night[order]
    cal_time = cal_time[order]
    corr_flux = corr_flux[order]

    # 未校准夜晚:与锚星不连通,或参照数据中根本没有该夜
    drop_nights = tgt["night"][~keep]
    drop_counts = Counter(str(n) for n in drop_nights)
    uncalibrated = []
    for night in sorted(drop_counts, key=_natural_key):
        uncalibrated.append({
            "night_id": night,
            "n_target_rows": int(drop_counts[night]),
            "reason": ("disconnected_from_anchor" if night in ref_night_set
                       else "not_in_reference"),
        })

    # 校正后的剩余观测必须满足原周期入口规则
    problems: list[str] = []
    detail: dict = {
        "n_target_rows": int(tgt["night"].size),
        "n_remaining": int(cal_time.size),
        "n_excluded": int((~keep).sum()),
        "uncalibrated_nights": uncalibrated,
    }
    if cal_time.size < MIN_ROWS:
        problems.append(
            f"扣除未校准夜晚后仅剩 {cal_time.size} 行,"
            f"少于原周期入口要求的 {MIN_ROWS} 行")
        detail["required_min_rows"] = MIN_ROWS
    if cal_time.size:
        unique, counts = np.unique(cal_time, return_counts=True)
        dup_times = unique[counts > 1]
        if dup_times.size:
            shown = "、".join(f"{float(v):.10g}" for v in dup_times[:8])
            problems.append(f"校正后存在重复观测时刻:{shown}")
            detail["duplicate_times"] = [float(v) for v in dup_times]
        if not np.all(np.isfinite(corr_flux)):
            problems.append("校正后通量存在非有限数值(请检查参照测量)")
            detail["nonfinite_flux"] = True
    if problems:
        raise CalibrationError(
            400, "target_entry_rules",
            "校正结果不满足原周期入口规则,未生成校正文件:" +
            ";".join(problems), detail)

    # 输出仅含 time_day、flux 两列,按时刻升序;repr 保留浮点原值
    lines = ["time_day,flux"]
    lines.extend(f"{float(t)!r},{float(f)!r}"
                 for t, f in zip(cal_time, corr_flux))
    csv_text = "\n".join(lines) + "\n"

    target_counts = Counter(str(n) for n in cal_night)
    nights_report = []
    for night in fit["used_nights"]:
        nights_report.append({
            "night_id": night,
            "throughput": throughputs[night],
            "n_reference_rows": int(fit["used_night_counts"].get(night, 0)),
            "n_target_rows": int(target_counts.get(night, 0)),
        })

    report = {
        "anchor_star_id": anchor,
        "anchor_flux": float(anchor_flux),
        "max_rel_dev": float(max_rel_dev),
        "n_reference_rows": int(ref["night"].size),
        "n_reference_used": fit["n_used"],
        "n_reference_excluded": fit["n_excluded"],
        "n_target_rows": int(tgt["night"].size),
        "n_target_calibrated": int(cal_time.size),
        "n_target_excluded": int((~keep).sum()),
        "nights": nights_report,
        "uncalibrated_nights": uncalibrated,
        "reference_star_fluxes": [
            {"star_id": s, "intrinsic_flux": fit["star_fluxes"][s]}
            for s in sorted(fit["star_fluxes"], key=_natural_key)
        ],
        "csv": csv_text,
    }
    return report
