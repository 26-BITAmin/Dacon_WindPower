# ═══════════════════════════════════════════════════════
# exp-108 진단: FiCR 단가 구간의 방향별 분해 지도 (그룹 × 실측 CF 구간 × 분기)
#
# 배경: exp-030이 "CF 0.7~0.9에서 손실 33%·71% 과소예측"을 지목했으나 그건 챔피언 이전
#   base 기준이고 방향·단가 구간이 섞여 있었다. 여기서는 **채점되는 값**(후처리 후 블렌드)을
#   FiCR 단가 경계(6%/8%)에 방향을 살려 6구간으로 쪼개고, 그룹·실측 CF 구간·분기로 지도를 그린다.
#
# 6구간 (signed error e = (pred − actual)/cap):
#   ① e < −8%      과소, 0원      ④ 0 ≤ e ≤ +6%   과대, 4원
#   ② −8 ≤ e < −6% 과소, 3원      ⑤ +6 < e ≤ +8%  과대, 3원
#   ③ −6 ≤ e < 0   과소, 4원      ⑥ e > +8%       과대, 0원
#
# ※ 연도 축은 불가: OOF 캐시가 2024년만 덮는다 (2022·2023은 폴드 없음, 2025는 라벨 없음).
#   따라서 시간 축은 분기로 대체.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp108_ficr_map.py
# ═══════════════════════════════════════════════════════
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp005_qopt import CAPACITY_KWH, DATA_DIR
from exp035_raw_dl import HOLD_FOLD
from exp043_post_resweep import postprocess_flex
from exp049_clean import PP_NEW
from exp060_pergroup_hpo import load_cache_full, build_slices

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

LAB = ["① 과소 >8% (0원)", "② 과소 6~8% (3원)", "③ 과소 0~6% (4원)",
       "④ 과대 0~6% (4원)", "⑤ 과대 6~8% (3원)", "⑥ 과대 >8% (0원)"]
COL = ["#8c1d18", "#d1665e", "#a8d5a2", "#5aa469", "#7fb3d5", "#1b4f72"]
CF_EDGES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
CF_LAB = ["0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", "0.5–0.6",
          "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9+"]


def build_tidy(col="blend"):
    cache, cb = load_cache_full()
    cache["blend"] = 0.65 * cache["cf_hpo"] + 0.35 * cache["dl_cln03"]
    parts = []
    for (fn, g), sub in build_slices(cache):
        if fn == HOLD_FOLD["name"]:            # 홀드아웃은 CV와 동일 행 → 중복 제외
            continue
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        p = postprocess_flex(cache[col].to_numpy()[sub.index], cap, sub["ws"].to_numpy(),
                             sub["pA"].to_numpy(), g, cb[fn], **PP_NEW)
        parts.append(sub[["group", "target", "forecast_kst_dtm"]].assign(pred=p, cap=cap))
    t = pd.concat(parts, ignore_index=True)
    t = t[t["target"].notna() & (t["target"] >= t["cap"] * 0.10)].copy()
    t["cf"] = t["target"] / t["cap"]
    t["e"] = (t["pred"] - t["target"]) / t["cap"]
    t["단가"] = np.select([t["e"].abs() <= 0.06, t["e"].abs() <= 0.08], [4.0, 3.0], default=0.0)
    t["b"] = np.select([t.e < -0.08, t.e < -0.06, t.e < 0, t.e <= 0.06, t.e <= 0.08],
                       [0, 1, 2, 3, 4], default=5)
    t["cfbin"] = pd.cut(t["cf"], CF_EDGES, labels=CF_LAB, right=False)
    t["q"] = t["forecast_kst_dtm"].dt.quarter
    return t


def ficr(s):
    return (s["target"] * s["단가"]).sum() / (s["target"] * 4).sum() if len(s) else np.nan


def main():
    t = build_tidy()
    print(f"평가 대상 {len(t):,}행 | 전체 FiCR {np.mean([ficr(t[t.group==g]) for g in (1,2,3)]):.4f}",
          flush=True)

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for gi, g in enumerate([1, 2, 3]):
        tg = t[t.group == g]
        fg = ficr(tg)
        denom = (tg["target"] * 4).sum()

        # ── 행1: 실측 CF 구간별 6구간 시간 비율 (stacked) ──
        ax = axes[0, gi]
        piv = (tg.groupby(["cfbin", "b"], observed=False).size().unstack(fill_value=0))
        piv = piv.reindex(columns=range(6), fill_value=0)
        share = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0) * 100
        bot = np.zeros(len(share))
        for b in range(6):
            ax.bar(range(len(share)), share[b], bottom=bot, color=COL[b],
                   label=LAB[b] if gi == 0 else None, width=0.78)
            bot += share[b].fillna(0).to_numpy()
        ax.set_xticks(range(len(share))); ax.set_xticklabels(share.index, rotation=45, fontsize=8)
        ax.set_ylim(0, 100); ax.set_ylabel("시간 비율 (%)" if gi == 0 else "")
        ax.set_title(f"그룹 {g}  (FiCR {fg:.3f}, n={len(tg):,})", fontsize=11, weight="bold")
        ax.axhline(50, color="k", lw=0.5, ls=":")
        for i, n in enumerate(piv.sum(axis=1)):
            ax.text(i, 101, f"{int(n)}", ha="center", fontsize=6, color="gray")

        # ── 행2: 실측 CF 구간별 FiCR 손실 기여 (그룹 내 정규화, 아래/위) ──
        ax = axes[1, gi]
        dn, up = [], []
        for lab in CF_LAB:
            s = tg[tg.cfbin == lab]
            d = s[s.b < 3]; u = s[s.b >= 3]
            dn.append((d["target"] * (4 - d["단가"])).sum() / denom)
            up.append((u["target"] * (4 - u["단가"])).sum() / denom)
        ax.bar(range(len(CF_LAB)), dn, color="#c0392b", label="과소 손실" if gi == 0 else None, width=0.78)
        ax.bar(range(len(CF_LAB)), up, bottom=dn, color="#2471a3",
               label="과대 손실" if gi == 0 else None, width=0.78)
        ax.set_xticks(range(len(CF_LAB))); ax.set_xticklabels(CF_LAB, rotation=45, fontsize=8)
        ax.set_ylabel("FiCR 손실 기여" if gi == 0 else "")
        ax.set_title(f"손실 합 {sum(dn)+sum(up):.3f} (과소 {sum(dn):.3f} / 과대 {sum(up):.3f})",
                     fontsize=10)
        if gi == 0:
            ax.legend(fontsize=8)

        # ── 행3: 분기별 FiCR + 손실 방향 ──
        ax = axes[2, gi]
        fs, dq, uq = [], [], []
        for q in [1, 2, 3, 4]:
            s = tg[tg.q == q]; dq_ = (s["target"] * 4).sum()
            fs.append(ficr(s))
            d = s[s.b < 3]; u = s[s.b >= 3]
            dq.append((d["target"] * (4 - d["단가"])).sum() / dq_)
            uq.append((u["target"] * (4 - u["단가"])).sum() / dq_)
        x = np.arange(4)
        ax.bar(x, dq, color="#c0392b", width=0.6, label="과소 손실" if gi == 0 else None)
        ax.bar(x, uq, bottom=dq, color="#2471a3", width=0.6, label="과대 손실" if gi == 0 else None)
        ax2 = ax.twinx()
        ax2.plot(x, fs, "o-", color="black", lw=2, ms=7, label="FiCR" if gi == 0 else None)
        ax2.set_ylim(0.30, 0.55); ax2.set_ylabel("FiCR" if gi == 2 else "")
        ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylim(0, 0.75); ax.set_ylabel("손실 (그룹·분기 내 정규화)" if gi == 0 else "")
        ax.set_title("분기별 (연도 축은 2024만 존재)", fontsize=10)
        if gi == 0:
            ax.legend(fontsize=8, loc="upper left")

    axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle("FiCR 단가 구간 방향별 분해 — 후처리 후 챔피언 블렌드(w0.65), 2024 CV 4폴드",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = DATA_DIR.parent / "exp108_ficr_map.png"
    fig.savefig(out, dpi=130)
    print(f"저장 → {out}", flush=True)

    # 콘솔 요약
    print("\n=== 그룹 × 실측 CF 구간: 과소 손실 비중 (%) ===", flush=True)
    rows = []
    for g in [1, 2, 3]:
        tg = t[t.group == g]; den = (tg["target"] * 4).sum()
        r = {}
        for lab in CF_LAB:
            s = tg[tg.cfbin == lab]
            d = s[s.b < 3]; u = s[s.b >= 3]
            dl = (d["target"] * (4 - d["단가"])).sum() / den
            ul = (u["target"] * (4 - u["단가"])).sum() / den
            r[lab] = f"{dl:.3f}/{ul:.3f}"
        rows.append(pd.Series(r, name=f"g{g} (과소/과대)"))
    print(pd.DataFrame(rows).to_string(), flush=True)


if __name__ == "__main__":
    main()
