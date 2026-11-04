"""离线计算单类旋转框 AP（VOC all-point），用于多缺失率评测。

为什么不用 DOTAMetric
---------------------
`DOTAMetric` 只吃 mmengine 的 DT 结构，要凭空插一个「换缺失率重评」的流程很别扭；
更要紧的是它**只出 mAP**，而本课题同时要 mAOE。既然 `mos_metrics` 已经有
IoU / 贪心匹配 / 角度误差，把 AP 也在这里算，一条链路就能同时给出
AP50 / AP75 / mAOE / 角度分位，且完全不依赖 mmengine 版本。

★ 正确性靠交叉验证保证，不靠自说自话：
  用本函数对 full 模型在 val 上的预测算 AP50/AP75，必须与日志里 DOTAMetric 报的
  0.3180 / 0.0910 吻合（实测见文件末的 __main__ 段）。
  对不上就说明这里的匹配口径和官方不一致，此时**不能**拿它出任何数字。

口径说明（写论文要注明的）：
- 单类别（ship），无 difficult 标记（MOS-Ship 第 10 字段一律当 0）。
- 每张图内预测按 score 降序贪心匹配未被占用的 GT，IoU>=thr 记为 TP。
- 跨图汇总后按 VOC2010+ 的 all-point 插值算 AP。
"""
import os
import os.path as osp

from mos_metrics import poly_iou, read_dota_dir


def compute_ap(preds, gts, iou_thr=0.5, score_thr=0.0, n_point=None):
    """返回 (ap, 细节 dict)。preds/gts 由 read_dota_dir 给出。

    n_point=None 用 VOC2010+ all-point；n_point=11 用 VOC2007 十一点插值。
    ★ 后者只为**核对** mmrotate 默认口径（`dota_metric.py` 未传 use_07_metric）
    而保留：同一条 PR 曲线上两种插值会给出不同的数，跨口径比较没有意义。
    本文正文数字一律取 n_point=None。
    """
    records = []
    n_gt_total = 0
    n_pred = 0
    for k in sorted(gts):
        gs = gts[k]
        n_gt_total += len(gs)
        ps = [p for p in preds.get(k, []) if p['score'] >= score_thr]
        ps.sort(key=lambda x: -x['score'])
        n_pred += len(ps)
        matched = [False] * len(gs)
        for p in ps:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gs):
                if matched[j]:
                    continue
                iou = poly_iou(p['poly'], g['poly'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= iou_thr
            if is_tp:
                matched[best_j] = True
            records.append((p['score'], is_tp))

    records.sort(key=lambda x: -x[0])
    tp = fp = 0
    prec, rec = [], []
    for _, is_tp in records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt_total if n_gt_total else 0.0)

    # VOC2010+ all-point interpolation
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    if n_point == 11:
        # VOC2007 十一点：在 recall = 0,0.1,...,1.0 处取"该召回之上的最大精度"
        ap = 0.0
        for k in range(11):
            r = k / 10.0
            p_max = 0.0
            for i in range(len(mrec)):
                if mrec[i] >= r:
                    p_max = max(p_max, mpre[i])
            ap += p_max / 11.0
    else:
        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

    return ap, dict(gt=n_gt_total, pred=n_pred, tp=tp,
                    recall=(tp / n_gt_total if n_gt_total else 0.0),
                    iou_thr=iou_thr)


def eval_dir(pred_dir, gt_dir, ious=(0.5, 0.75), score_thr=0.0):
    """对导出的预测目录算一组 AP。返回 {thr: (ap, detail)}。"""
    preds = read_dota_dir(pred_dir)
    gts = read_dota_dir(gt_dir)
    common = set(preds) & set(gts)
    if not common:
        raise ValueError(f'预测与 GT 无同名文件: {pred_dir} vs {gt_dir}')
    out = {}
    for t in ious:
        out[t] = compute_ap(preds, gts, iou_thr=t, score_thr=score_thr)
    return out


if __name__ == '__main__':
    import sys
    from mos_metrics import evaluate_dirs

    if len(sys.argv) < 3:
        print('usage: python eval_ap.py <pred_dir> <gt_dir> [score_thr]')
        raise SystemExit(1)
    pred, gt = sys.argv[1], sys.argv[2]
    score_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    r = eval_dir(pred, gt, score_thr=score_thr)
    print('pred_dir = %s\ngt_dir   = %s' % (pred, gt))
    for t, (ap, d) in r.items():
        print(f'  AP@{t}: {ap:.4f}  gt={d["gt"]} pred={d["pred"]} '
              f'tp={d["tp"]} recall={d["recall"]:.4f}')
    m = evaluate_dirs(pred, gt, iou_thr=0.5, score_thr=score_thr)
    print(f'  mAOE@0.5 = {m["mAOE"]:.4f} deg (n={m["n"]})')
