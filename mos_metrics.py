# -*- coding: utf-8 -*-
"""
mos_metrics.py —— 旋转框的角度误差指标（mAOE / 误差分位 / 漏检翻转）

★★ 为什么这个文件必须存在 ★★
    实测 MMRotate 1.x 的 DOTAMetric 源码里写死 ``allowed_metrics = ['mAP']``，
    传别的 metric 直接 KeyError。**框架不提供 mAOE，也不会提供。**
    而本课题的研究空白②正是「mAP 掩盖角度退化」——如果连角度误差都没人算，
    这个空白就无从证明。所以指标得自己写。

★★ 一个必须先说清的定义问题 ★★
    MOS-Ship / DOTA 的标注是四顶点。转成 (cx, cy, w, h, θ) 时，le90 约定要求
    **w 是长边、θ ∈ [-90°, 90°)**。这个约定本身把「180° 翻转」抹掉了：
    船头朝东的框和船头朝西的框，在 le90 下是**同一个数**。

    后果有两条，写论文时必须分开讲：
      (1) mAOE（长边方位误差）能反映「框贴合得歪不歪」，但**看不见航向翻转**。
      (2) 要测航向，必须换成保留方向的编码（按顶点顺序的 le135，或角度分类头），
          否则「翻转率」这个指标在 le90 下**定义不出来**，硬报就是假数据。

    因此本文件给的是 mAOE + 误差分位数（p50/p90）+ 大误差占比，
    并明确不提供 le90 下的翻转率——下面 ``orientation_flip_rate()`` 只在
    你传入非 le90 的原始角度时才可用，且会校验这一点，防止被误用。

用法（命令行）：
    python mos_metrics.py --selftest                     # 合成用例，验证几何与角度逻辑
    python mos_metrics.py --real /path/to/MOS-Ship   # 在真实标注上回环校验 + 报角度分布
"""

import argparse
import math
import os
import sys

# ======================================================================
#  一、基本几何
# ======================================================================
def signed_area(poly):
    """鞋带公式。正负号表示绕向（图像坐标 y 向下时，正号对应视觉上的顺时针）。"""
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a * 0.5


def poly_area(poly):
    return abs(signed_area(poly))


def _seg_inter(p1, p2, a, b):
    """线段 p1p2 与**直线** ab 的交点。理论上线段一定跨越，故无需再判范围。"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:                      # 平行/退化：返回端点，宁可误差一点也不要崩
        return p1
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _clip_convex(subject, clip):
    """Sutherland–Hodgman：用凸多边形 clip 裁 subject。两者都必须是凸的（旋转框天然是）。"""
    if len(subject) < 3 or len(clip) < 3:
        return []
    if signed_area(clip) < 0:                 # 统一成同一绕向，否则内部/外部判断会反
        clip = clip[::-1]
    out = list(subject)
    n = len(clip)
    for i in range(n):
        a = clip[i]
        b = clip[(i + 1) % n]
        ax, ay = a
        bx, by = b

        def inside(p, ax=ax, ay=ay, bx=bx, by=by):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) >= 0

        if not out:
            return []
        inp = out
        out = []
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(_seg_inter(s, e, a, b))
                out.append(e)
            elif inside(s):
                out.append(_seg_inter(s, e, a, b))
            s = e
    return out


def poly_iou(p1, p2):
    """凸多边形 IoU。"""
    a1 = poly_area(p1)
    a2 = poly_area(p2)
    if a1 < 1e-12 or a2 < 1e-12:
        return 0.0
    inter = _clip_convex(p1, p2)
    ai = poly_area(inter) if len(inter) >= 3 else 0.0
    union = a1 + a2 - ai
    return 0.0 if union < 1e-12 else max(0.0, min(1.0, ai / union))


# ======================================================================
#  二、表示转换
# ======================================================================
def rbox_to_poly(cx, cy, w, h, theta_deg):
    """(cx,cy,w,h,θ°) → 四顶点，顺序与 DOTA 一致（左上起、顺时针）。"""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    hw, hh = w * 0.5, h * 0.5
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    out = []
    for dx, dy in corners:
        out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return out


def poly_to_rbox_le90(poly):
    """
    四顶点 → (cx, cy, w, h, θ)，**le90 约定：w 为长边，θ ∈ [-90°, 90°)**。

    步骤：
      1. 中心取四顶点均值（对任意平行四边形都成立，比求对角线交点稳）
      2. 边长取对边平均，抗标注抖动
      3. 取较长那组边的方向作为 θ
      4. 规范化：θ 折进 [-90,90)；若 w<h 则交换并把 θ 抬 90° 再折一次

    第 4 步就是「长边约定」的落地，也是 le90 丢失 180° 航向信息的地方。
    """
    if len(poly) != 4:
        raise ValueError("需要 4 个顶点")
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = poly
    cx = (x0 + x1 + x2 + x3) / 4.0
    cy = (y0 + y1 + y2 + y3) / 4.0

    e1 = (x1 - x0, y1 - y0)                   # 边 0->1
    e2 = (x2 - x1, y2 - y1)                   # 边 1->2
    e3 = (x3 - x2, y3 - y2)                   # 边 2->3
    e4 = (x0 - x3, y0 - y3)                   # 边 3->0

    l1 = 0.5 * (math.hypot(*e1) + math.hypot(*e3))
    l2 = 0.5 * (math.hypot(*e2) + math.hypot(*e4))

    # 用「较长的边对」定方向。取两条同向边的平均向量，避免单边噪声。
    if l1 >= l2:
        ux, uy = 0.5 * (e1[0] - e3[0]), 0.5 * (e1[1] - e3[1])
        w, h = l1, l2
    else:
        ux, uy = 0.5 * (e2[0] - e4[0]), 0.5 * (e2[1] - e4[1])
        w, h = l2, l1
    theta = math.degrees(math.atan2(uy, ux))

    def fold(a):
        a = math.fmod(a, 180.0)               # → (-180, 180)
        if a >= 90.0:
            a -= 180.0
        elif a < -90.0:
            a += 180.0
        return a

    theta = fold(theta)
    if w < h:                                 # 长边约定
        w, h = h, w
        theta = fold(theta + 90.0)
    return cx, cy, w, h, theta


def dota8_to_rbox_le90(xy):
    """DOTA 的 8 个数 → le90 rbox。"""
    return poly_to_rbox_le90([(xy[i], xy[i + 1]) for i in range(0, 8, 2)])


# ======================================================================
#  三、角度误差
# ======================================================================
def angle_diff(pred_deg, gt_deg):
    """
    两个 le90 角之差，折进 (-90, 90]，取绝对值的**有符号原值**。

    为什么折 180 而不是 360：le90 下 θ 与 θ+180 是同一个框，
    所以差值的周期是 180°，折进 (-90, 90] 后 |Δ| ≤ 90。
    """
    d = (pred_deg - gt_deg) % 180.0           # [0, 180)
    if d > 90.0:
        d -= 180.0                            # → (-90, 90]
    return d


def angle_error(pred_deg, gt_deg):
    return abs(angle_diff(pred_deg, gt_deg))


def _quantile(sorted_vals, q):
    """线性插值分位。n=0 时返回 0.0。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ======================================================================
#  四、匹配与 mAOE
# ======================================================================
def match_greedy(preds, gts, iou_thr=0.5):
    """
    按分数从高到低贪心匹配，返回 [(pred_idx, gt_idx, iou), ...]。
    这是 DOTA / COCO 之外的简化版；单类别、无 crowd、够用。

    preds / gts 每项是 dict：
        {'poly': [(x,y)×4], 'rbox': (cx,cy,w,h,θ), 'score': float}
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i].get("score", 1.0))
    used = set()
    pairs = []
    for pi in order:
        # 用严格大于，平局时保留**先遇到的**那个 GT。
        # 早先写成 v >= best_iou，IoU 相同的候选会一路覆盖，最后总是配到最后一条 GT——
        # 在多目标场景下会让误差统计出现莫名其妙的偏移，且不易察觉。
        best_j, best_iou = -1, -1.0
        for gj in range(len(gts)):
            if gj in used:
                continue
            v = poly_iou(preds[pi]["poly"], gts[gj]["poly"])
            if v > best_iou:
                best_j, best_iou = gj, v
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            pairs.append((pi, best_j, best_iou))
    return pairs


def aoe_stats(preds, gts, iou_thr=0.5):
    """
    在 IoU≥iou_thr 的匹配对上统计角度误差。

    返回 dict：
        n          匹配对数（= 参与角度统计的 TP 数）
        mAOE       平均绝对方位误差（度）——这就是要报的 mAOE
        p50/p90    误差分位数
        frac_gt30  误差 >30° 的占比；均值会掩盖双峰分布，这个指标才是「退化」的证据
        max        最大误差
        recall     匹配对数 / GT 总数（便于和 mAP 一起看待）

    注意：只统计匹配上的框。「漏检」不计入 mAOE（漏掉的船没有角度预测），
    但 recall 一并返回，防止「靠多漏检换低 mAOE」这种自欺。
    """
    pairs = match_greedy(preds, gts, iou_thr)
    errs = []
    for pi, gj, _ in pairs:
        e = angle_error(preds[pi]["rbox"][4], gts[gj]["rbox"][4])
        errs.append(e)
    errs.sort()
    n = len(errs)
    return {
        "n": n,
        "mAOE": (sum(errs) / n) if n else float("nan"),
        "p50": _quantile(errs, 0.50),
        "p90": _quantile(errs, 0.90),
        "frac_gt30": (sum(1 for e in errs if e > 30.0) / n) if n else float("nan"),
        "max": errs[-1] if errs else float("nan"),
        "recall": (n / len(gts)) if gts else float("nan"),
        "iou_thr": iou_thr,
    }


def orientation_flip_rate(preds, gts, iou_thr=0.5, lo=120.0, hi=180.0001,
                          angle_convention="le90"):
    """
    航向翻转率：匹配上的框里，预测角与 GT 角相差接近 180° 的比例。

    判据：d = |wrap(pred - gt)| ∈ [lo, hi]，默认 [120°, 180°]。
    窗口必须**含 180°**——完全反向是最该被计数的情形。
    （早先默认写成 [60,120]，恰好把 180° 排除在外，等于漏掉所有真正的翻转。）

    ★ 这个指标只在**保方向**的编码下才有定义。★
      默认参数 ``angle_convention='le90'``，此时本函数**直接抛错**。
      原因是 le90 把 w 定为长边、θ 折进 [-90°,90°)，一个框旋转 180° 之后
      数值完全不变 —— 翻转在数据里已被抹平，算出来的 0 是「指标失效」，
      不是「模型完美」。若把这个 0 写进论文，就是拿一个不可能有值的指标充数。

      要真测航向，得换保方向编码：
        'le135'  长边 135 约定，θ ∈ [-45°,135°)，仍保留 180° 区分
        'oc'     OpenCV 约定，θ ∈ (0°,90°]，同样保留
        'vertex' 直接依赖标注的顶点顺序（DOTA 的四点有固定绕向，能表达首尾）

      本函数不去猜输入是哪种编码 —— 猜错会静默给出错的数，比报错危险得多。
      所以要求调用方显式声明。
    """
    if angle_convention == "le90":
        raise ValueError(
            "le90 下 180° 翻转不可观测（旋转 180° 数值不变），"
            "此指标无定义。请改用 angle_convention='le135' / 'oc' / 'vertex'，"
            "或改用基于顶点顺序的方向判定。")
    if angle_convention not in ("le135", "oc", "vertex", "raw"):
        raise ValueError(f"未知的 angle_convention: {angle_convention}")

    pairs = match_greedy(preds, gts, iou_thr)
    if not pairs:
        return float("nan")
    flips = 0
    for pi, gj, _ in pairs:
        tp = preds[pi]["rbox"]
        tg = gts[gj]["rbox"]
        d = abs((tp[4] - tg[4] + 180.0) % 360.0 - 180.0)     # → [0,180]
        if lo <= d <= hi:
            flips += 1
    return flips / len(pairs)


# ======================================================================
#  五、目录级离线评测（不依赖 mmengine）
# ======================================================================
def read_dota_dir(d, prefix=""):
    """
    读一个目录下所有 DOTA 格式 txt，返回 {key: [rbox, ...]}。
    文件名 <key>.txt（可带 prefix 后缀，如 val 目录里的 _rgb）。
    """
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".txt"):
            continue
        key = f[:-4]
        if prefix:
            key = key[:-len(prefix)] if key.endswith(prefix) else key
        boxes = []
        with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                p = ln.split()
                if len(p) < 9:
                    continue
                try:
                    xy = [float(v) for v in p[:8]]
                except ValueError:
                    continue
                rb = dota8_to_rbox_le90(xy)
                boxes.append({"poly": [(xy[i], xy[i + 1]) for i in range(0, 8, 2)],
                              "rbox": rb,
                              "score": float(p[8]) if len(p) > 8 and _isfloat(p[8]) else 1.0})
        out[key] = boxes
    return out


def _isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def evaluate_dirs(pred_dir, gt_dir, iou_thr=0.5, score_thr=0.0,
                  pred_prefix="", gt_prefix=""):
    """
    对一个目录的预测结果做 mAOE 评测。GT 用同一 key 的标注文件。

    为什么要这个入口：MMRotate 的 DOTAMetric 只出 mAP，而且它内部会把结果写成
    json 或打包成提交 zip，中途想插一个角度指标很别扭。更省事的做法是让推理脚本
    直接按 DOTA 格式（8 坐标 + 分数 + 类别）落盘，然后用这里离线评测——
    既能算 mAOE，又完全不依赖 mmengine 版本，换框架也不用改。
    """
    preds_all = read_dota_dir(pred_dir, pred_prefix)
    gts_all = read_dota_dir(gt_dir, gt_prefix)
    common = sorted(set(preds_all) & set(gts_all))
    if not common:
        raise ValueError(f"预测与 GT 没有同名文件：{pred_dir} vs {gt_dir}")

    errs = []
    n_gt = n_pred = 0
    for key in common:
        ps = [p for p in preds_all[key] if p["score"] >= score_thr]
        gs = gts_all[key]
        n_pred += len(ps)
        n_gt += len(gs)
        for pi, gj, _ in match_greedy(ps, gs, iou_thr):
            errs.append(angle_error(ps[pi]["rbox"][4], gs[gj]["rbox"][4]))
    errs.sort()
    n = len(errs)
    return {
        "images": len(common),
        "gt": n_gt, "pred": n_pred, "n": n,
        "mAOE": (sum(errs) / n) if n else float("nan"),
        "p50": _quantile(errs, 0.50),
        "p90": _quantile(errs, 0.90),
        "frac_gt30": (sum(1 for e in errs if e > 30.0) / n) if n else float("nan"),
        "max": errs[-1] if errs else float("nan"),
        "recall_at_iou": (n / n_gt) if n_gt else float("nan"),
        "iou_thr": iou_thr,
    }

# ======================================================================
#  六、自检与真实数据校验
# ======================================================================
def selftest(verbose=True):
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            if verbose:
                print(f"  [PASS] {name}")
        else:
            fail += 1
            print(f"  [FAIL] {name}  {extra}")

    def close(a, b, tol=1e-6):
        return abs(a - b) <= tol

    print("-" * 70)
    print("  A. 几何：面积与 IoU")
    print("-" * 70)
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    chk("单位方块面积 = 100", close(poly_area(sq), 100.0), poly_area(sq))
    chk("自身 IoU = 1", close(poly_iou(sq, sq), 1.0), poly_iou(sq, sq))
    chk("平移 20 无交 → IoU = 0",
        close(poly_iou(sq, [(20, 0), (30, 0), (30, 10), (20, 10)]), 0.0))
    # 右移 5：交 50，并 150 → 1/3
    chk("半重叠 IoU = 1/3",
        close(poly_iou(sq, [(5, 0), (15, 0), (15, 10), (5, 10)]), 1 / 3, 1e-9),
        poly_iou(sq, [(5, 0), (15, 0), (15, 10), (5, 10)]))
    # 与自身绕向相反的多边形，IoU 仍应为 1（绕向处理）
    chk("反向绕向仍 IoU=1", close(poly_iou(sq, sq[::-1]), 1.0), poly_iou(sq, sq[::-1]))
    # 旋转 45° 的正方形，边长 10 → 若与原正方形同心，其外接… 用面积关系验证裁剪不崩
    rot = rbox_to_poly(5, 5, 10, 10, 45)
    v = poly_iou(sq, rot)
    chk("旋转 45° 交并比在 (0,1) 内", 0.0 < v < 1.0, v)

    print("-" * 70)
    print("  B. 角度：le90 规范化与差值")
    print("-" * 70)
    # 一个 20×5 的横条，θ=0
    p0 = rbox_to_poly(0, 0, 20, 5, 0)
    cx, cy, w, h, th = poly_to_rbox_le90(p0)
    chk("横条 → w=20,h=5,θ=0", close(w, 20) and close(h, 5) and close(th, 0),
        (w, h, th))
    # 竖条（w=5,h=20,θ=0）本质上和横条转向 90° 一样 → 规范化后应仍是 w=20,h=5,θ=±90
    p1 = rbox_to_poly(0, 0, 5, 20, 0)
    cx, cy, w, h, th = poly_to_rbox_le90(p1)
    chk("竖条被规范化成 w≥h", w >= h - 1e-9 and close(w, 20), (w, h, th))
    chk("竖条 θ 落在 [-90,90)",
        -90.0 - 1e-9 <= th <= 90.0 + 1e-9 and (close(abs(th), 90) or close(th, 0)),
        th)

    # 同一个矩形用「换边 + 转 90°」表示，规范化后应完全一致（这是 le90 的核心不变量）
    a = poly_to_rbox_le90(rbox_to_poly(3, 4, 30, 8, 17))
    b = poly_to_rbox_le90(rbox_to_poly(3, 4, 8, 30, 17 + 90))
    same = (close(a[2], b[2], 1e-6) and close(a[3], b[3], 1e-6)
            and close(a[4], b[4], 1e-6))
    chk("换边+90° 与 原表示 规范化后相同", same, (a, b))

    chk("角度差 0", close(angle_error(10, 10), 0.0))
    chk("角度差 1", close(angle_error(11, 10), 1.0))
    chk("角度差 89", close(angle_error(99, 10), 89.0), angle_error(99, 10))
    chk("角度差 91 → 折成 89", close(angle_error(101, 10), 89.0),
        angle_error(101, 10))
    chk("角度差 179 → 折成 1", close(angle_error(189, 10), 1.0),
        angle_error(189, 10))
    chk("跨 ±90 边界（-88 与 88 → 4）", close(angle_error(-88, 88), 4.0),
        angle_error(-88, 88))
    chk("差值恒 ≤ 90", all(angle_error(a_, b_) <= 90.0 + 1e-9
                          for a_ in range(-180, 181, 7)
                          for b_ in range(-180, 181, 11)))

    print("-" * 70)
    print("  C. mAOE 聚合")
    print("-" * 70)
    def mk(rb, score=1.0):
        return {"poly": rbox_to_poly(*rb), "rbox": rb, "score": score}

    gts = [mk((100, 100, 40, 12, 0.0))]
    preds = [mk((100, 100, 40, 12, 5.0))]
    s = aoe_stats(preds, gts)
    chk("单个预测偏 5° → mAOE=5", close(s["mAOE"], 5.0, 1e-6), s)

    preds = [mk((500, 500, 40, 12, 0.0))]        # 完全错位
    s = aoe_stats(preds, gts)
    chk("无匹配 → n=0 且 recall=0",
        s["n"] == 0 and close(s["recall"], 0.0), s)

    # ---- 聚合逻辑用「同一多边形 + 只改声明的 θ」隔离测试 ----------------------
    # 早先的版本让 pred 与 gt 各用不同角度跑匹配，40° 差在 40×12 的细长框上 IoU 低于 0.5、
    # 匹配不上，测试反而挂了 —— 那是把「匹配」与「聚合」混在一处测。
    # 这里进一步让**所有 GT 的 θ 都取 0**：无论配对怎样落，误差集合都只由 pred 自己的 θ 决定，
    # 于是期望值是配对无关的，测试才有唯一解。
    poly3 = rbox_to_poly(100, 100, 40, 12, 0.0)
    gts = [{"poly": poly3, "rbox": (100, 100, 40, 12, 0.0), "score": 1.0}
           for _ in range(3)]
    preds = [{"poly": poly3, "rbox": (100, 100, 40, 12, t), "score": 1.0}
             for t in (0.0, 5.0, 40.0)]
    s = aoe_stats(preds, gts)
    # 误差集合恒为 {0, 5, 40} → 均值 15；有序表上 p50=5，p90 = 5*0.2 + 40*0.8 = 33
    chk("3 个目标全匹配", s["n"] == 3, s)
    chk("mAOE = (0+5+40)/3 = 15", close(s["mAOE"], 15.0, 1e-6), s["mAOE"])
    chk("p50 = 5", close(s["p50"], 5.0, 1e-6), s["p50"])
    chk("p90 = 33", close(s["p90"], 33.0, 1e-6), s["p90"])
    chk("frac_gt30 = 1/3", close(s["frac_gt30"], 1 / 3, 1e-9), s["frac_gt30"])
    chk("max = 40", close(s["max"], 40.0, 1e-6), s["max"])
    chk("recall = 1.0", close(s["recall"], 1.0, 1e-9), s["recall"])

    print("-" * 70)
    print("  D. le90 下翻转率必须拒绝计算（防止报假指标）")
    print("-" * 70)
    try:
        orientation_flip_rate(preds, gts)                 # 默认就是 le90
        chk("le90 输入应抛错", False, "居然没抛错，会静默报 0")
    except ValueError as e:
        chk("le90 输入抛 ValueError", "不可观测" in str(e), e)
    try:
        orientation_flip_rate([], [], angle_convention="随便写的")
        chk("未知约定应抛错", False, "居然没抛错")
    except ValueError as e:
        chk("未知约定抛 ValueError", "未知" in str(e), e)
    # 保方向约定下：175° 是接近完全反向的翻转，必须被算出来
    fp = [{"poly": poly3, "rbox": (100, 100, 40, 12, 175.0), "score": 1.0}]
    fg = [{"poly": poly3, "rbox": (100, 100, 40, 12, 0.0), "score": 1.0}]
    chk("le135 下 175° 记为翻转",
        close(orientation_flip_rate(fp, fg, angle_convention="le135"), 1.0),
        orientation_flip_rate(fp, fg, angle_convention="le135"))
    chk("le135 下 180° 记为翻转",
        close(orientation_flip_rate(
            [{"poly": poly3, "rbox": (100, 100, 40, 12, 180.0), "score": 1.0}],
            fg, angle_convention="le135"), 1.0))
    chk("le135 下 10° 不算翻转",
        close(orientation_flip_rate(
            [{"poly": poly3, "rbox": (100, 100, 40, 12, 10.0), "score": 1.0}],
            fg, angle_convention="le135"), 0.0))
    chk("le135 下 100° 不算翻转（不足 120° 门限）",
        close(orientation_flip_rate(
            [{"poly": poly3, "rbox": (100, 100, 40, 12, 100.0), "score": 1.0}],
            fg, angle_convention="le135"), 0.0))

    print("-" * 70)
    print("  E. 目录级评测端到端（人为偏转已知角度，检验能否算回原值）")
    print("-" * 70)
    import tempfile

    def write_dota(path, poly, score):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(" ".join("%.2f" % v for p in poly for v in p)
                     + " ship %.3f\n" % score)

    with tempfile.TemporaryDirectory() as td:
        gt_d = os.path.join(td, "gt")
        pr_d = os.path.join(td, "pred")
        os.makedirs(gt_d)
        os.makedirs(pr_d)
        # 长宽比控制在 2~3，避免几度偏转就把 IoU 压到 0.5 以下（那会测成「漏检」而非「角度」）
        base = [(100, 100, 200, 80, 10.0),
                (400, 300, 180, 90, -30.0),
                (700, 500, 220, 100, 45.0)]
        delta = [5.0, 0.0, 12.0]
        for i, (cx, cy, w, h, th) in enumerate(base):
            write_dota(os.path.join(gt_d, "img%d.txt" % i),
                       rbox_to_poly(cx, cy, w, h, th), 1.0)
            write_dota(os.path.join(pr_d, "img%d.txt" % i),
                       rbox_to_poly(cx, cy, w, h, th + delta[i]), 0.9)
        res = evaluate_dirs(pr_d, gt_d)
        # 容差取 1e-2 而不是 1e-6：坐标是以 "%.2f" 落盘的，0.005px 的量化误差
        # 经 arctan 放大后约为 1e-3 度量级（实测 12.0005 vs 12）。
        # 这也顺带量出了一个实感：DOTA 常见的一位小数坐标，对 mAOE 的影响在 0.0x° 量级，
        # 可以忽略——但前提是别用三位小数去要求它。
        chk("读回 3 图 3 框并全部匹配", res["n"] == 3, res)
        chk("mAOE = (5+0+12)/3 = 5.667",
            close(res["mAOE"], 17.0 / 3.0, 1e-2), res["mAOE"])
        chk("max = 12", close(res["max"], 12.0, 1e-2), res["max"])
        chk("recall_at_iou = 1.0", close(res["recall_at_iou"], 1.0), res["recall_at_iou"])

    print("-" * 70)
    print(f"  自检结果：{ok} 通过 / {fail} 失败")
    print("-" * 70)
    return fail == 0


def real_check(root, n_files=0, verbose=True):
    """
    真实数据回环校验 + 角度分布交叉验证。

    两件事：
      1) DOTA 8 点 → le90 rbox → 8 点，原多边形与还原多边形的 IoU 应接近 1。
         这一步能同时验掉「多边形解析」「中心/边长估计」「方向折叠」三处实现错误。
      2) 把 le90 角度反推回原始 [0,180) 口径，按 15° 分 12 桶，
         与 tools/mos_ship_stats.py 独立产出的 angle_hist 逐桶比对。
         两套互不相干的代码如果对得上，角度解析才算真的可信。

    n_files=0 表示**扫描全部标注文件**。
    ★ 别用「排在前面的 N 个文件」当抽样：文件名以场景号开头，
      sorted() 取前 80 个实际上只覆盖了一个港区，角度分布会严重失真
      （实测单场景偏差可达 20 个百分点）。
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mos_core import parse_dota

    ann_dirs = [os.path.join(root, "train", "labelTxt"),
                os.path.join(root, "val", "rgb_labelTxt"),
                os.path.join(root, "val", "sar_labelTxt")]
    files = []
    for d in ann_dirs:
        if os.path.isdir(d):
            fs = sorted(os.listdir(d))
            if n_files and n_files > 0:
                # 均匀间隔取，覆盖不同场景；比取头部或纯随机更稳
                step = max(1, len(fs) // n_files)
                fs = fs[::step][:n_files]
            files += [os.path.join(d, f) for f in fs]
    if not files:
        print("[error] 没找到标注目录，检查 --real 是否指向 MOS-Ship 根目录")
        return False

    ious = []
    angles = []
    bad = []
    for fp in files:
        for xy, _cat in parse_dota(fp):
            cx, cy, w, h, th = dota8_to_rbox_le90(xy)
            back = rbox_to_poly(cx, cy, w, h, th)
            orig = [(xy[i], xy[i + 1]) for i in range(0, 8, 2)]
            v = poly_iou(orig, back)
            ious.append(v)
            angles.append(th)
            if v < 0.99:
                bad.append((os.path.basename(fp), round(v, 4)))

    n = len(ious)
    ious_s = sorted(ious)
    print(f"\n  真实标注回环：{len(files)} 个文件, {n} 个框")
    print(f"    IoU 均值 {sum(ious)/n:.6f}  最小 {ious_s[0]:.6f}  "
          f"p1 {_quantile(ious_s, 0.01):.6f}")
    print(f"    IoU < 0.99 的框：{len(bad)} 个")
    for x in bad[:5]:
        print("      -", x)

    # ---- le90 直方图 ----
    buckets = [0] * 12
    for th in angles:
        idx = int((th + 90.0) // 15.0)
        buckets[max(0, min(11, idx))] += 1
    print("\n  GT 长边角度分布（le90，15° 一桶，从 -90° 起）")
    for i, c in enumerate(buckets):
        lo = -90 + i * 15
        bar = "#" * int(60.0 * c / max(buckets))
        print(f"    [{lo:>4d},{lo+15:>4d})  {c:5d}  {100.0*c/n:5.1f}%  {bar}")

    # ---- 反推回 [0,180) 口径，与既有统计交叉验证 ----
    raw = [0] * 12
    for th in angles:
        r = th if th >= 0 else th + 180.0          # le90 → [0,180)
        raw[max(0, min(11, int(r // 15.0)))] += 1
    print("\n  反推回原始 [0,180) 口径（用于与 mos_ship_stats.py 的 angle_hist 比对）")
    print("    " + "  ".join(f"{v:5d}" for v in raw))

    ref = _load_reference_hist(root)
    hist_ok = True
    if ref is not None:
        ref_t, total = ref
        if total == n:
            diff = [abs(a - b) for a, b in zip(raw, ref_t)]
            hist_ok = sum(diff) <= max(3, int(0.002 * n))
            print(f"    参考值 {total} 个框：" + "  ".join(f"{v:5d}" for v in ref_t))
            print(f"    逐桶最大偏差 {max(diff)}，合计偏差 {sum(diff)} → "
                  f"{'一致' if hist_ok else '不一致，需查口径'}")
        else:
            print(f"    框数不同（本次 {n} vs 参考 {total}），跳过比对")
    else:
        print("    （未找到参考统计 json，跳过比对）")

    ok_n = len(ious) - len(bad)
    all_ok = (not bad) and hist_ok
    print(f"\n  回环校验：{'全部通过' if not bad else str(ok_n) + '/' + str(n) + ' 通过'}")
    print(f"  角度交叉验证：{'通过' if hist_ok else '未通过'}")
    return all_ok


def _load_reference_hist(root):
    """读 tools/mos_ship_stats_summary.json 里各划分的 angle_hist，求和作为参考值。"""
    import json
    for cand in (os.path.join(os.path.dirname(os.path.dirname(
                     os.path.abspath(__file__))), "tools",
                     "mos_ship_stats_summary.json"),):
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        tot = [0] * 12
        for s in d.get("splits", []):
            h = s.get("angle_hist") or []
            if len(h) == 12:
                for i in range(12):
                    tot[i] += h[i]
        if sum(tot) > 0:
            return tot, sum(tot)
    return None


def main():
    ap = argparse.ArgumentParser(description="旋转框角度指标（mAOE）自检与真实数据校验")
    ap.add_argument("--selftest", action="store_true", help="跑合成用例")
    ap.add_argument("--real", default=None, help="MOS-Ship 根目录，做回环校验")
    ap.add_argument("--files", type=int, default=0,
                    help="每个标注目录取多少个文件，0=全部（默认，避免单场景抽样偏差）")
    args = ap.parse_args()

    rc = 0
    if args.selftest or not args.real:
        print("=" * 70)
        print("  mos_metrics 自检")
        print("=" * 70)
        if not selftest():
            rc = 1
    if args.real:
        print()
        print("=" * 70)
        print("  真实标注回环校验")
        print("=" * 70)
        if not real_check(args.real, args.files):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
