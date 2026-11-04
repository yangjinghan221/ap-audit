"""扫描「11 点插值」与「all-point 插值」随召回水平变化的偏差方向。

背景：2026-11-01 发现 11 点 AP 与 all-point AP 的差不是固定的。同为一份预测，
召回 0.952 时 11 点低 3.85 点，召回 0.426 时 11 点反高 2.86 点。若把任一读数当
「真值」写进论文，方向都可能写反。本脚本用「按分数全局截断、只保留前 N 个预测」
造出连续可控的召回序列，把这条偏差曲线画出来。

用法 / Usage:
    python scan_ap_interp.py <pred_dir> <gt_dir> [IoU]
"""
import sys

from eval_ap import compute_ap
from mos_metrics import read_dota_dir

# 覆盖「几乎无预测 → 全量」的截断档位
KEEP = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 648, 800, 1000, 1200]


def truncate(preds, n):
    """按 score 全局降序只保留前 n 个框，其余图保留空列表（键必须留着）。"""
    flat = []
    for img, bs in preds.items():
        for b in bs:
            flat.append((b['score'], img, b))
    flat.sort(key=lambda t: -t[0])
    keep = flat[:n]
    out = {img: [] for img in preds}
    for _, img, b in keep:
        out[img].append(b)
    for img in out:
        out[img].sort(key=lambda b: -b['score'])
    return out


def main():
    pred_dir, gt_dir = sys.argv[1], sys.argv[2]
    iou = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    preds = read_dota_dir(pred_dir)
    gts = read_dota_dir(gt_dir)
    common = set(preds) & set(gts)
    n_gt = sum(len(gts[k]) for k in common)
    preds = {k: v for k, v in preds.items() if k in common}
    n_all = sum(len(v) for v in preds.values())
    print(f'pred_dir = {pred_dir}')
    print(f'GT 框 {n_gt} 个，预测 {n_all} 个，IoU={iou}\n')
    print(f'{"保留":>6} {"召回":>7} {"all-point":>10} {"11-point":>9} '
          f'{"11−all":>8}')
    rows = []
    # 把「全量不截断」也作为最后一档，否则最后一行会与单次运行的结果混用
    for n in sorted(set(KEEP) | {n_all}):
        if n > n_all:
            continue
        sub = truncate(preds, n)
        ap_all, d = compute_ap(sub, gts, iou_thr=iou, score_thr=0.0)
        ap_11, _ = compute_ap(sub, gts, iou_thr=iou, score_thr=0.0,
                              n_point=11)
        rows.append((n, d['recall'], ap_all, ap_11))
        print(f'{n:>6} {d["recall"]:>7.4f} {ap_all:>10.4f} {ap_11:>9.4f} '
              f'{ap_11 - ap_all:>+8.4f}')

    if len(rows) >= 2:
        lo = min(rows, key=lambda r: r[3] - r[2])
        hi = max(rows, key=lambda r: r[3] - r[2])
        print(f'\n11 点最低估：保留 {lo[0]} 个（召回 {lo[1]:.4f}）'
              f' {lo[3] - lo[2]:+.4f}')
        print(f'11 点最高估：保留 {hi[0]} 个（召回 {hi[1]:.4f}）'
              f' {hi[3] - hi[2]:+.4f}')
        sign_change = any((a[3] - a[2]) * (b[3] - b[2]) < 0
                          for a, b in zip(rows, rows[1:]))
        print(f'偏差符号是否随召回翻转：{"是" if sign_change else "否"}')


if __name__ == '__main__':
    main()
