# -*- coding: utf-8 -*-
"""
mos_core.py —— MOS-Ship 数据层核心（**纯标准库，不依赖 torch/numpy**）

职责：
    1. 由 patch 名解析出 4 个文件的真实路径（train 与 val 两种目录布局都支持）
    2. 解析 DOTA 风格标注（8 坐标 + 类别 + 第 10 字段）
    3. 生成「模态缺失」掩码，**以 patch 为单位、可复现、按协议可配置**
    4. 强制一条规则：模态不可用时，该模态的标注也一并不返回

为什么单独抽出来：
    训练环境（Linux + torch）与体检环境（你这台 Windows、没装 torch）都要用它；
    纯标准库保证 `python check_data.py` 在哪都能跑。

标注格式提醒（2026-10-25 实测）：
    x1 y1 x2 y2 x3 y3 x4 y4  ship  <f10>
    · 坐标是切片局部坐标；约 10% 实例有负坐标或越出 1024 边界（框不截断）
    · 第 10 字段**不是 difficult**：1 是可重复的默认值，非 1 值仅保证单文件内唯一。
      → 一律忽略，全部实例按 difficult=0 处理（若按 !=1 过滤会丢掉 58%）。
"""

import hashlib
import os

DEFAULT_ROOT = os.environ.get("MOS_ROOT", "data/MOS-Ship")
TILE = 1024
IMG_EXT = ".png"


# ------------------------------------------------------------------ 路径解析
def resolve_paths(root, patch):
    """
    返回 dict：
        rgb, sar          图像绝对路径
        rgb_lbl, sar_lbl  标注绝对路径（**两模态各自独立**）
        cloud             合成云光学图（仅 val 目录的 patch 有，否则 None）
        src               'train' 或 'val'（本 patch 原本落在哪个官方目录）
    """
    out = {"rgb": None, "sar": None, "rgb_lbl": None, "sar_lbl": None,
           "cloud": None, "src": None}

    # 布局一：train/images + train/labelTxt
    p = {
        "rgb": os.path.join(root, "train", "images", f"{patch}_rgb{IMG_EXT}"),
        "sar": os.path.join(root, "train", "images", f"{patch}_sar{IMG_EXT}"),
        "rgb_lbl": os.path.join(root, "train", "labelTxt", f"{patch}_rgb.txt"),
        "sar_lbl": os.path.join(root, "train", "labelTxt", f"{patch}_sar.txt"),
    }
    if all(os.path.isfile(v) for v in p.values()):
        out.update(p)
        out["src"] = "train"
        return out

    # 布局二：val/rgb + val/sar + val/{rgb,sar}_labelTxt（文件名同样带后缀）
    q = {
        "rgb": os.path.join(root, "val", "rgb", f"{patch}_rgb{IMG_EXT}"),
        "sar": os.path.join(root, "val", "sar", f"{patch}_sar{IMG_EXT}"),
        "rgb_lbl": os.path.join(root, "val", "rgb_labelTxt", f"{patch}_rgb.txt"),
        "sar_lbl": os.path.join(root, "val", "sar_labelTxt", f"{patch}_sar.txt"),
    }
    if all(os.path.isfile(v) for v in q.values()):
        out.update(q)
        out["src"] = "val"
        c = os.path.join(root, "val", "rgb_clouds", f"{patch}_rgb{IMG_EXT}")
        out["cloud"] = c if os.path.isfile(c) else None
        return out

    return out                 # 全 None，调用方负责报错


def missing_files(paths):
    return [k for k in ("rgb", "sar", "rgb_lbl", "sar_lbl") if not paths.get(k)]


# ------------------------------------------------------------------ 标注解析
def parse_dota(path):
    """
    返回 [(xy8, cat), ...]，xy8 为 8 个 float。
    第 10 字段（f10）直接丢弃——它既不是 difficult 也不是稳定 ID。
    """
    if not path or not os.path.isfile(path):
        return []
    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                text = fh.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except OSError:
            return []
    if text is None:
        return []

    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.lower().startswith("imagesource"):
            continue
        p = ln.split()
        if len(p) < 9:
            continue
        try:
            xy = [float(v) for v in p[:8]]
        except ValueError:
            continue
        out.append((xy, p[8]))
    return out


# ------------------------------------------------------------------ 可见比例
def _clip_poly(poly, w, h):
    """Sutherland–Hodgman，裁剪到 [0,w]×[0,h]。"""
    def clip(poly, inside, inter):
        out = []
        if not poly:
            return out
        s = poly[-1]
        for e in poly:
            if inside(e):
                if not inside(s):
                    out.append(inter(s, e))
                out.append(e)
            elif inside(s):
                out.append(inter(s, e))
            s = e
        return out

    def area(p):
        a = 0.0
        for i in range(len(p)):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % len(p)]
            a += x1 * y2 - x2 * y1
        return abs(a) * 0.5

    p = poly
    p = clip(p,
             lambda q: q[0] >= 0,
             lambda a, b: (0.0, a[1] + (b[1] - a[1]) * (0 - a[0]) / (b[0] - a[0])))
    p = clip(p,
             lambda q: q[0] <= w,
             lambda a, b: (float(w), a[1] + (b[1] - a[1]) * (w - a[0]) / (b[0] - a[0])))
    p = clip(p,
             lambda q: q[1] >= 0,
             lambda a, b: (a[0] + (b[0] - a[0]) * (0 - a[1]) / (b[1] - a[1]), 0.0))
    p = clip(p,
             lambda q: q[1] <= h,
             lambda a, b: (a[0] + (b[0] - a[0]) * (h - a[1]) / (b[1] - a[1]), float(h)))
    return p, area(p)


def visible_ratio(xy, w=TILE, h=TILE):
    """框落在图内的面积占比。DOTA 里越界框不截断，训练时可据此过滤/加权。"""
    poly = [(xy[i], xy[i + 1]) for i in range(0, 8, 2)]
    a0 = 0.0
    for i in range(4):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % 4]
        a0 += x1 * y2 - x2 * y1
    a0 = abs(a0) * 0.5
    if a0 < 1e-9:
        return 0.0
    _, a1 = _clip_poly(poly, w, h)
    return min(1.0, a1 / a0)


# ------------------------------------------------------------------ 缺失协议
def _u01(*parts):
    """稳定的 [0,1) 伪随机数：同一输入永远同一输出（跨进程、跨机器一致）。"""
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


class MissingProtocol:
    """
    模态缺失协议。**施加单位是 patch，不是目标**（每 patch 中位仅 1 个目标，
    目标级丢弃会让 mAP 分母漂移，不同缺失率之间不可比）。

    支持：
        full()                                  两模态全给（上界参考）
        smr(p)                                  共享缺失率：各模态独立以 p 丢弃
        imr_f(p_opt=(lo,hi), p_sar=(lo,hi))      物理不对称：缺失率由场景决定
                                                （模拟云量气候 / 过境覆盖，非随机）
        imr_e(p_opt, p_sar)                     极端不对称（如 0.8 / 0.1）
        custom(p_opt, p_sar)                    固定常数

    ensure_one=False（默认）：
        两模态独立抽样，**丢弃率严格等于设定值**——论文里写 p=0.5，实现就必须是 0.5。
        代价是会出现「两路都缺」的样本（p=0.5 时约 25%，p=0.75 时约 56%），
        这类样本没有任何监督信号，目标检测里会被当成纯负样本污染训练。
        处理方式是**在数据集层把它们排除出索引**（见 mos_dataset.MoSShipDataset.active），
        并把「有效样本数」如实报出来 —— 而不是偷偷改抽样概率去凑数。
        若置 True，则会重抽补回一路，但**实际丢弃率会低于设定值**（p=0.75 时实测只有 0.42），
        只在「必须保证每张图都有监督」的实验里用，且论文中必须说明真实丢弃率。
    """

    def __init__(self, kind="full", p_opt=0.0, p_sar=0.0, seed=0,
                 ensure_one=False):
        self.kind = kind
        self.seed = seed
        self.ensure_one = ensure_one
        if kind == "smr":
            self.p_opt = self.p_sar = float(p_opt)
        elif kind == "imr_f":
            lo, hi = p_opt if isinstance(p_opt, (tuple, list)) else (p_opt, p_opt)
            lo2, hi2 = p_sar if isinstance(p_sar, (tuple, list)) else (p_sar, p_sar)
            self.range_opt = (float(lo), float(hi))
            self.range_sar = (float(lo2), float(hi2))
        elif kind in ("imr_e", "custom"):
            self.p_opt = float(p_opt)
            self.p_sar = float(p_sar)
        elif kind != "full":
            raise ValueError("未知协议: %s" % kind)

    def rates(self, scene):
        """返回该场景下的 (p_opt, p_sar)。"""
        if self.kind == "full":
            return 0.0, 0.0
        if self.kind in ("smr", "imr_e", "custom"):
            return self.p_opt, self.p_sar
        # imr_f：场景级的固定偏移，而不是每次随机 —— 这样同一场景的缺失是「一致的」，
        # 更接近真实（一个地区一段时期云量高就一直高）
        f = _u01("scene", self.seed, scene)
        p_opt = self.range_opt[0] + f * (self.range_opt[1] - self.range_opt[0])
        g = _u01("scene_sar", self.seed, scene)
        p_sar = self.range_sar[0] + g * (self.range_sar[1] - self.range_sar[0])
        return p_opt, p_sar

    def availability(self, patch, scene, epoch=0):
        """
        返回 (avail_rgb, avail_sar)。
        epoch 参与哈希：训练时每轮换一批掩码（等价于数据增广），
        评测时固定 epoch=0，保证所有模型面对完全相同的缺失模式——这是公平比较的前提。
        """
        p_opt, p_sar = self.rates(scene)
        a_rgb = _u01("opt", self.seed, epoch, patch) >= p_opt
        a_sar = _u01("sar", self.seed, epoch, patch) >= p_sar
        if self.ensure_one and not (a_rgb or a_sar):
            # 重抽方向：按各自缺失率反比，缺得多的那一路优先补回
            if _u01("fix", self.seed, epoch, patch) < 0.5:
                a_rgb = True
            else:
                a_sar = True
        return a_rgb, a_sar

    def materialize(self, patches, epoch=0):
        """
        对给定 patch 列表生成掩码，**丢弃数精确等于 round(p × n)**。

        做法：按 (salt, seed, epoch, patch) 的哈希值排序，取最低的 k 个丢弃。
        因为排序只依赖哈希、与列表顺序无关，所以：
            · 结果完全可复现（换机器、换进程一致）
            · 实际丢弃率 = k/n，与设定值精确相等，不存在 ±2% 的抽样偏差
            · imr_f 按场景分组计算，保证每个场景内部的缺失率也精确
        返回 {patch: (avail_rgb, avail_sar)}

        注意：不保证「至少留一路」。两路都缺的样本由数据集层排除出索引。
        """
        by_scene = {}
        for p in patches:
            by_scene.setdefault(p.split("_")[0], []).append(p)

        out = {}
        for sc, ps in by_scene.items():
            p_opt, p_sar = self.rates(sc)
            for idx, (salt, p) in enumerate((("opt", p_opt), ("sar", p_sar))):
                n = len(ps)
                k = int(round(p * n))
                order = sorted(ps, key=lambda q: _u01(salt, self.seed, epoch, q))
                drop = set(order[:k])
                for q in ps:
                    v = out.setdefault(q, [True, True])
                    v[idx] = (q not in drop)
        return {q: (v[0], v[1]) for q, v in out.items()}

    def __repr__(self):
        if self.kind == "full":
            return "MissingProtocol(full)"
        if self.kind == "imr_f":
            return "MissingProtocol(imr_f, opt=%s, sar=%s, seed=%s)" % (
                self.range_opt, self.range_sar, self.seed)
        return "MissingProtocol(%s, p_opt=%.2f, p_sar=%.2f, seed=%s)" % (
            self.kind, self.p_opt, self.p_sar, self.seed)


# ------------------------------------------------------------------ 样本组装
def load_sample(root, patch, protocol=None, epoch=0, min_visible=0.0,
                use_cloud=False, avail=None):
    """
    组装一个 patch 的样本（纯 Python 结构，不读像素、不转 tensor）。

    avail: 显式指定 (avail_rgb, avail_sar)。给了就用它，不再问 protocol。
           数据集层用 protocol.materialize() 算出的精确掩码从这里传进来，
           避免「数据集用精确掩码、这里又按阈值重算」导致两边不一致。

    返回 dict：
        patch, scene, src
        avail          {'rgb': bool, 'sar': bool}
        boxes          {'rgb': [[8 floats],...], 'sar': [...]}
                        —— **模态不可用时对应列表为空**，这是「丢模态连标注一起丢」的落地点
        labels         {'rgb': ['ship',...], 'sar': [...]}
        visible        {'rgb': [ratio,...], 'sar': [...]}
        paths          可选图像路径（rgb 若被丢弃，训练时根本不去读它）
    """
    scene = patch.split("_")[0]
    if protocol is None:
        protocol = MissingProtocol("full")
    if avail is None:
        a_rgb, a_sar = protocol.availability(patch, scene, epoch)
    else:
        a_rgb, a_sar = avail
    paths = resolve_paths(root, patch)

    out = {"patch": patch, "scene": scene, "src": paths.get("src"),
           "avail": {"rgb": bool(a_rgb), "sar": bool(a_sar)},
           "boxes": {"rgb": [], "sar": []},
           "labels": {"rgb": [], "sar": []},
           "visible": {"rgb": [], "sar": []},
           "paths": {}}

    for mod, avail, key in (("rgb", a_rgb, "rgb"),
                            ("sar", a_sar, "sar")):
        if not avail:
            continue                       # ← 图像不读、标注不读，一路到底
        img = paths.get(key)
        if use_cloud and mod == "rgb" and paths.get("cloud"):
            img = paths["cloud"]
        out["paths"][mod] = img
        for xy, cat in parse_dota(paths.get(f"{key}_lbl")):
            vis = visible_ratio(xy)
            if vis < min_visible:
                continue
            out["boxes"][mod].append(xy)
            out["labels"][mod].append(cat)
            out["visible"][mod].append(vis)
    return out


def split_summary(root, patches, protocol, epoch=0):
    """统计一个划分在某协议下的可用性，用于核对协议是否按预期工作。"""
    n = len(patches)
    masks = protocol.materialize(patches)
    c = {"patches": n, "rgb": 0, "sar": 0, "both": 0, "none": 0,
         "rgb_boxes": 0, "sar_boxes": 0}
    for p in patches:
        s = load_sample(root, p, protocol, epoch, avail=masks[p])
        ar, as_ = s["avail"]["rgb"], s["avail"]["sar"]
        c["rgb"] += ar
        c["sar"] += as_
        c["both"] += ar and as_
        c["none"] += (not ar and not as_)
        c["rgb_boxes"] += len(s["boxes"]["rgb"])
        c["sar_boxes"] += len(s["boxes"]["sar"])
    c["drop_rgb"] = round(1 - c["rgb"] / max(1, n), 4)
    c["drop_sar"] = round(1 - c["sar"] / max(1, n), 4)
    return c
