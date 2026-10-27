"""扫描件清理：去彩色噪点、去黑边、方向检测与旋转校正、淡字加深、去页边噪点。

处理顺序：
  0. 去除大面积浅色彩色噪点（粉色墨粉斑、彩色竖条纹等，颜色不写死，保留印章、红头）
  1. 去除贴边黑边、黑角与页边“边框带”（按原始扫描坐标处理）
  2. 方向检测（0/90/180/270°，Tesseract OSD），把文字转到朝上
  3. 小角度倾斜校正，旋转后露出的空角用周围纸张纹理修补
  4. 正文淡字加深（只加深成行的文字笔画，不加深噪点、污渍、阴影；本来就清楚的字不动），
     很淡的字再做细笔画插值，把扫成一串断点的笔画连成连续的线
  5. 去除版心之外的零星噪点，以及版心内远离文字、表格线的墨粉散点（正文行的笔画不动）

“宁可不转，也不错转”：方向需在四个旋转副本上检测一致、且与文字行方向不矛盾才旋转；
倾斜角需峰值明显、两种分辨率下结果一致才校正；保持原尺寸，内容会转出画布时整体平移，
只允许本来就被图片边缘截断的墨迹再转出去一点，否则不校正。
除 90°/270° 方向旋转会交换宽高外，输出尺寸与原图一致。

用法: python pic.py "图片路径或目录" [-t 阈值] [-m 扩展像素] [--no-rotate] [--no-enhance] [--no-interp] [--no-denoise] [-n 噪点阈值] [-v]
结果保存到 ./output 目录，文件名与原图相同。
"""
import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

OUTPUT_DIR = Path(__file__).parent / "output"
log = logging.getLogger("pic")

# 噪点阈值默认值：孤立墨点的最大尺寸占典型字高的比例
NOISE_RATIO = 0.43

# 彩色噪点：Lab 色度（相对纸张底色）高于 COLOR_MIN 算“带颜色”，高于 COLOR_STRONG 算浓色内容（红头、印章）
COLOR_MIN = 6.0
COLOR_STRONG = 45.0
# 只清除与噪点主色相相差不超过该角度的颜色
COLOR_HUE_TOL = 35.0
# 印章保护：色度高于该值的像素连成有厚度的块即视为印章（骑缝章颜色较浅，阈值低于 COLOR_STRONG）
STAMP_CHROMA = 25.0
# 大片浅色彩色区域里，文字笔画（成行、边缘锐利）占带色像素的比例不低于它，就是浅色的彩色字（如淡红头），不清除
COLOR_TEXT_FRAC = 0.4
# 骑缝章外圈检测：半径占图片短边的范围（A4 上 38~45mm 的公章约 0.09~0.11）、
# 图内可见圆弧的最小角度、可见圆弧上有墨迹的最低比例
STAMP_R_MIN = 0.05
STAMP_R_MAX = 0.15
STAMP_ARC_MIN_DEG = 40.0
STAMP_ARC_COVER = 0.6
STAMP_ARC_RUN = 4  # 圆周按 2° 采样，连续命中 4 个采样点（8°）才算一段圆弧

# 方向检测：OSD 输入图的最长边；每次检测的最低置信度
OSD_MAX_SIDE = 2000
OSD_MIN_CONF = 1.0
# 文字行方向比值（行投影锐度 / 列投影锐度，>1 为横排）：
# 90°/270° 旋转后若低于 ROTATE_VETO，说明旋转后文字成了竖排，否决旋转；
# 方向未经 OSD 确认且低于 VERTICAL_TEXT 时，文字可能是横躺的，跳过去噪避免把正文当噪点
ROTATE_VETO = 0.45
VERTICAL_TEXT = 1.0

# 倾斜校正：搜索范围、最小校正角、相对 0° 的最小增益、相对远处次峰的最小峰值比、两种分辨率允许的偏差
SKEW_RANGE = 15.0
SKEW_MIN = 0.2
SKEW_MIN_GAIN = 1.1
SKEW_MIN_PEAK = 1.3
SKEW_AGREE = 0.3

# 文字增强：黑帽核直径占短边的比例（比笔画宽、比阴影窄）；笔画暗度的最低门槛；
# 找行用的边缘梯度门槛与边缘密度门槛；
# 正文行的最低边缘锐度（梯度 90 分位）与最低笔画暗度（95 分位）；加深后的目标墨色灰度
ENHANCE_STROKE = 0.0125
ENHANCE_MIN_DARK = 3.0
ENHANCE_EDGE = 15.0
ENHANCE_EDGE_DENSITY = 0.03
ENHANCE_SHARP = 35.0
ENHANCE_LINE_DARK = 20.0
ENHANCE_INK = 30.0
# 附近笔画暗度（黑帽值）低于 ENHANCE_FAINT 的是淡字，完全加深；高于 ENHANCE_CLEAR 的字本来就清楚，不动
ENHANCE_FAINT = 70.0
ENHANCE_CLEAR = 140.0
ENHANCE_BASE_TOL = 12.0  # 局部底色比纸色暗超过它（阴影、色块）就不加深
# 附近笔画暗度（黑帽值）低于 FAINT 才完全插值，高于 CLEAR 不插值
ENHANCE_INTERP_FAINT = 45.0
ENHANCE_INTERP_CLEAR = 80.0


def imread(path: Path) -> np.ndarray | None:
    # np.fromfile + imdecode 以支持中文路径
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"无法编码图片: {path}")
    buf.tofile(str(path))


def dominant_color(img: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """内容区域内占比最大的颜色（量化到 8 级一档后取众数，再取该档均值）。"""
    pixels = img[mask > 0]
    if len(pixels) == 0:
        return 255, 255, 255
    keys = (pixels // 8).astype(np.int32)
    flat = keys[:, 0] * 1024 + keys[:, 1] * 32 + keys[:, 2]
    top = np.bincount(flat).argmax()
    b, g, r = (int(v) for v in pixels[flat == top].mean(axis=0))
    return b, g, r


def stamp_mask(img: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """骑缝章（压在页边、被图片边缘截断的印章）的保护掩码：任何清除步骤都不得改动其中的像素。

    骑缝章颜色往往很浅、又紧贴图片边缘，容易被当成黑边过渡带或贴边残迹清掉。两条线索取并集：
      - 颜色：贴边、有厚度的带色块（彩色扫描）；
      - 形状：贴边的残缺圆弧，即印章外圈（黑白扫描时颜色线索失效，靠它识别）。
    页面内部的印章不贴边，本来就不会被清除，这里不管。
    """
    H, W = img.shape[:2]
    keep = cv2.bitwise_or(color_stamp_mask(img, bg), arc_stamp_mask(img, bg))
    if keep.any():
        r = max(3, int(0.01 * min(H, W)))
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        log.info("  印章保护: %.2f%% 的面积", 100 * float(np.count_nonzero(keep)) / keep.size)
    return keep


def color_stamp_mask(img: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """贴边的彩色印章块。

    按 Lab 色度（相对纸张底色）找带色像素，闭运算把印章的笔画连成整块，只保留贴边、有厚度、有面积的块；
    黑边、直线旁扫描产生的 1~3px 色差彩边很细，不会被当成印章。
    """
    H, W = img.shape[:2]
    side = min(H, W)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(np.full((1, 1, 3), bg, np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    chroma = np.hypot(lab[..., 1] - paper_lab[1], lab[..., 2] - paper_lab[2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 黑边里偏色的暗像素不算（印章再浓也不会接近黑色）
    core = ((chroma > STAMP_CHROMA) & (gray > float(np.mean(bg)) * 0.35)).astype(np.uint8)
    keep = np.zeros((H, W), np.uint8)
    if not core.any():
        return keep
    # 印章的字之间有空隙：大核闭运算把同一枚印章的笔画连成整块
    k = max(5, int(0.03 * side) | 1)
    blob = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_b, st, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    # 厚度用距离变换衡量（L 形的细彩边外框也很大，但内部最厚处只有几个像素）
    depth = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    thick, min_px, edge = max(3.0, 0.004 * side), max(30, int(0.00003 * H * W)), max(3, int(0.02 * side))
    found = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if not (x <= edge or y <= edge or x + w >= W - edge or y + h >= H - edge):
            continue
        sel = lab_b[y:y + h, x:x + w] == i
        cnt = int(core[y:y + h, x:x + w][sel].sum())
        if float(depth[y:y + h, x:x + w][sel].max()) >= thick and cnt >= min_px:
            found[i] = True
            log.debug("    贴边彩色印章 (%d,%d) %dx%d（带色像素 %d）", x, y, w, h, cnt)
    if not found.any():
        return keep
    # 同一枚印章里离得稍远的零散笔画（较淡的偏旁、印边的残笔）也一并保护
    r = max(5, int(0.05 * side))
    near = cv2.dilate(found[np.asarray(lab_b, np.int32)].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    hit = np.zeros(n, bool)
    hit[np.unique(np.asarray(lab_b, np.int32)[(np.asarray(near) > 0) & (np.asarray(blob) > 0)])] = True
    hit[0] = False
    return hit[np.asarray(lab_b, np.int32)].astype(np.uint8) * 255


def arc_stamp_mask(img: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """被图片边缘截断的印章外圈（残缺圆弧）所围的圆形区域，不依赖颜色。

    把四条边依次转到右侧，在贴边条带内对“细线状”的墨迹做 RANSAC 拟合圆：
      - 半径在印章的常见范围内（STAMP_R_MIN~STAMP_R_MAX × 短边），圆被右边缘截断；
      - 圆周在图片内的可见部分至少 STAMP_ARC_MIN_DEG 度，其中 ≥ STAMP_ARC_COVER 被连续的墨迹段覆盖；
      - 圆周外侧、内侧各一圈几乎没有墨迹（是一条孤立的线，而不是成片的文字、手写、黑边）。
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keep = np.zeros((H, W), np.uint8)
    for t in (0, 90, 180, 270):  # 依次把上/右/下/左边转到右侧处理
        g = rotate_cw(gray, t)
        m = _arc_on_right(g, float(np.mean(bg)), min(H, W), t)
        if m is not None:
            keep |= rotate_cw(m, (360 - t) % 360)
    return keep


def _arc_on_right(g: np.ndarray, paper: float, side: int, t: int) -> np.ndarray | None:
    h, w = g.shape
    rmin, rmax = STAMP_R_MIN * side, STAMP_R_MAX * side
    x_off = max(0, w - int(2.2 * rmax))  # 只看贴右边的条带，圆整个落在条带内
    sub = g[:, x_off:]
    sw = sub.shape[1]
    ink = (sub < paper - 40).astype(np.uint8)
    # 去掉成块的黑边、黑角，只留线状墨迹（印章外圈是 1~2mm 宽的线）
    k = max(9, int(0.02 * side) | 1)
    ink &= cv2.bitwise_not(cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)),
                                      np.ones((5, 5), np.uint8))) // 255
    # 去掉表格线、黑边带等长直线：圆与直线相切时会重叠一段，造成假的“连续圆弧”。
    # 外圈在这个长度上弯曲得明显（偏离直线远超线宽），不会被去掉
    L = max(15, int(0.1 * side))
    straight = cv2.bitwise_or(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)),
                              cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8)))
    ink &= 1 - cv2.dilate(straight, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(ink)
    if xs.size < 50:
        return None
    rng = np.random.default_rng(0)  # 固定种子，结果可复现
    if xs.size > 20000:
        idx = rng.choice(xs.size, 20000, replace=False)
        xs, ys = xs[idx], ys[idx]
    tol = max(2.0, 0.004 * side)
    dist = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)

    # 随机取三点求外接圆
    N = 30000
    i = rng.integers(0, xs.size, (3, N))
    x1, y1, x2, y2, x3, y3 = (v.astype(np.float64) for v in (xs[i[0]], ys[i[0]], xs[i[1]], ys[i[1]], xs[i[2]], ys[i[2]]))
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    ok = np.abs(d) > 1e-6
    d = np.where(ok, d, 1.0)
    s1, s2, s3 = x1 ** 2 + y1 ** 2, x2 ** 2 + y2 ** 2, x3 ** 2 + y3 ** 2
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = np.hypot(x1 - cx, y1 - cy)
    # 被右边缘截断：圆超出右边缘，且图内至少露出 1/5 个直径
    ok &= (r >= rmin) & (r <= rmax) & (cx + r > sw + tol) & (cx <= sw + 0.6 * r) & (cx - r >= 0)
    ok &= (cy > -0.5 * r) & (cy < h + 0.5 * r)
    if not ok.any():
        return None
    cx, cy, r = cx[ok], cy[ok], r[ok]

    def score(cx, cy, r):
        ang = np.radians(np.arange(0, 360, 2.0))
        px = cx[:, None] + r[:, None] * np.cos(ang)
        py = cy[:, None] + r[:, None] * np.sin(ang)
        vis = (px >= 0) & (px < sw - 1) & (py >= 0) & (py < h - 1)
        pxi, pyi = np.clip(px, 0, sw - 1).astype(int), np.clip(py, 0, h - 1).astype(int)
        nvis = vis.sum(1)
        hit = (dist[pyi, pxi] <= tol) & vis
        # 只算连成长段（≥ STAMP_ARC_RUN 个采样点，即 8°）的命中：外圈是连续的弧，手写笔画只是零星穿过圆周
        L = STAMP_ARC_RUN
        ext = np.concatenate([hit, hit[:, :L - 1]], axis=1).astype(np.int32)  # 首尾相接，处理跨 0° 的弧
        c = np.concatenate([np.zeros((len(cx), 1), np.int32), np.cumsum(ext, axis=1)], axis=1)
        full = (c[:, L:] - c[:, :-L]) == L  # 从第 j 个采样点起连续 L 个都命中
        run = np.zeros_like(hit)
        for k in range(L):
            run |= np.roll(full, k, axis=1)
        cover = (run & vis).sum(1) / np.maximum(nvis, 1)
        def ring(dr: float) -> np.ndarray:  # 半径偏移 dr 处一圈上有墨迹的比例
            qx = np.clip(cx[:, None] + (r[:, None] + dr) * np.cos(ang), 0, sw - 1).astype(int)
            qy = np.clip(cy[:, None] + (r[:, None] + dr) * np.sin(ang), 0, h - 1).astype(int)
            return ((ink[qy, qx] > 0) & vis).sum(1) / np.maximum(nvis, 1)
        # 外圈是一条孤立的线：外侧是空白，内侧到印文之间也有一圈空隙；手写字、文字串会同时落在两侧
        return nvis * 2.0, cover, np.maximum(ring(3 * tol), ring(-2.5 * tol))

    span, cover, side_ink = score(cx, cy, r)
    good = (span >= STAMP_ARC_MIN_DEG) & (cover >= STAMP_ARC_COVER) & (side_ink <= 0.15)
    if not good.any():
        return None
    # 印章里的内圈、环形排列的印文也能拟合出圆，但它们两侧还有墨迹；外圈两侧最干净，优先取外圈
    b = int(np.argmax(np.where(good, (cover - 2 * side_ink) * span, -np.inf)))
    cx0, cy0, r0 = float(cx[b]), float(cy[b]), float(r[b])
    # 用圆周附近的墨迹点做最小二乘精修（代数拟合）
    near = np.abs(np.hypot(xs - cx0, ys - cy0) - r0) <= 2 * tol
    if near.sum() >= 10:
        A = np.c_[2 * xs[near], 2 * ys[near], np.ones(near.sum())]
        sol, *_ = np.linalg.lstsq(A, xs[near] ** 2.0 + ys[near] ** 2.0, rcond=None)
        c1, c2 = float(sol[0]), float(sol[1])
        r1 = float(np.sqrt(max(sol[2] + c1 ** 2 + c2 ** 2, 0)))
        sp1, cv1, _ = score(np.array([c1]), np.array([c2]), np.array([r1]))
        if rmin <= r1 <= rmax and cv1[0] >= cover[b]:
            cx0, cy0, r0 = c1, c2, r1
    # 印章圈内大部分是纸，只有印文笔画；成片的灰色阴影、手印的边沿也可能弯成一段圆弧，但圈内整片发灰
    inner = np.zeros((h, sw), np.uint8)
    cv2.circle(inner, (int(round(cx0)), int(round(cy0))), max(1, int(round(r0 - tol))), 1, -1)
    fill = float((sub[inner > 0] < paper - 25).mean()) if inner.any() else 0.0
    if fill > 0.5:
        log.debug("    贴边圆弧（边 %d°）圈内 %.0f%% 发灰，是阴影或污渍的边沿，不是印章", t, 100 * fill)
        return None
    log.debug("    贴边残缺圆弧（边 %d°）：圆心 (%.0f,%.0f) 半径 %.0f，可见 %.0f°，覆盖 %.2f，两侧 %.2f，圈内发灰 %.0f%%",
              t, cx0 + x_off, cy0, r0, span[b], cover[b], side_ink[b], 100 * fill)
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(round(cx0 + x_off)), int(round(cy0))), int(round(r0 + 2 * tol)), 255, -1)
    return m


def edge_dark_seed(gray: np.ndarray, bg: tuple[int, int, int], thresh: int) -> np.ndarray:
    """与图片边缘相连的深色连通域，加上最外圈几个像素里偏暗的残留：黑边的“种子”。"""
    H, W = gray.shape
    dark = (gray < thresh).astype(np.uint8)
    # 黑边是厚实的色块；伸到图片边缘的表格线、与之相交的笔迹都是细线，开运算后消失，不能作种子，
    # 否则整张表格线网连同签名会被当成一个贴边连通域整块清掉
    k = max(11, int(min(H, W) * 0.005) | 1)
    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    min_area = max(30, int(H * W * 0.00005))

    # 最外圈几个像素几乎一定是扫描边缘，偏暗的残留一并清掉
    rim = gray < int(np.mean(bg)) - 40
    seed = np.zeros((H, W), np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (x == 0 or y == 0 or x + w == W or y + h == H) and area >= min_area:
            seed[lab == i] = 255
    # 沿边缘走向的长细线才是扫描边缘残留；横穿边缘被截断的字（如序号“10”的“1”）、表格线头不算，
    # 被截成竖条的“1”也只有一个字高，比扫描残线短得多。四条边分开看，免得在页角连成一圈
    d = max(6, int(0.01 * min(H, W)))
    for m, depth in ((dark > 0, d), (rim, 4)):
        for t in (0, 90, 180, 270):  # 依次把四条边转到左侧
            sm = rotate_cw(m.astype(np.uint8), t)
            h_, w_ = sm.shape
            strip = np.zeros_like(sm)
            strip[:, :depth] = sm[:, :depth]
            n, lab, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
            out = np.zeros_like(sm)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if x == 0 and h >= max(3 * w, 0.03 * h_):
                    out[lab == i] = 255
            seed |= rotate_cw(out, (360 - t) % 360)
    return seed


def trim_keep_at_border(keep: np.ndarray, img: np.ndarray, bg: tuple[int, int, int], thresh: int,
                        margin: int) -> np.ndarray:
    """从印章保护区里扣掉压在章上的那一段黑边带。

    黑边沿整条边延伸，在保护区内的厚度按保护区两侧（保护区之外）测得的黑边厚度插值；
    只扣掉紧贴边缘的这一条，被黑边盖住的那点印章本来就看不见了，章的其余部分照常保护。
    """
    if not keep.any():
        return keep
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    seed = edge_dark_seed(gray, bg, thresh)
    band = np.zeros_like(keep)
    for t in (0, 90, 180, 270):  # 依次把四条边转到右侧处理
        sd, kp = rotate_cw(seed, t) > 0, rotate_cw(keep, t) > 0
        h, w = sd.shape
        under = kp[:, -1]  # 保护区压到右边缘的那些行
        if not under.any() or under.all():
            continue
        # 每行从右边缘起连续的深色像素数，即这一行黑边的厚度
        run = np.where(sd[:, ::-1].all(axis=1), w, np.argmin(sd[:, ::-1], axis=1)).astype(np.float64)
        rows = np.arange(h)
        est = np.interp(rows[under], rows[~under], run[~under])
        est = np.where(est > 0, est + 2 * margin, 0)  # 加上黑边外侧的灰色过渡
        m = np.zeros((h, w), np.uint8)
        m[rows[under][:, None], np.arange(w)[None, :]] = (np.arange(w)[None, :] >= w - est[:, None]) * 255
        band |= rotate_cw(m, (360 - t) % 360)
    if band.any():
        log.debug("    保护区内压着黑边带 %d 像素，照常清除", int(np.count_nonzero(band & keep)))
    return cv2.bitwise_and(keep, cv2.bitwise_not(band))


def charlike_ink(gray: np.ndarray, bg: tuple[int, int, int], region: np.ndarray,
                 exclude: np.ndarray | None = None, touch_edge: bool = True,
                 max_size: int | None = None) -> np.ndarray:
    """region 里“像字”的墨迹块（面积够大、不是细长线条）的保护掩码（外扩 2px）。

    exclude: 这些像素不算墨迹；touch_edge=False 时贴着图片边缘的块不保护（多半是被截断的残迹）；
    max_size: 外框长边超过它的块（长横线、表格框）不保护。
    """
    H, W = gray.shape
    ink = gray < int(np.mean(bg)) * 0.65
    if exclude is not None:
        ink &= exclude == 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    protect = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        line_like = min(w, h) <= 6 and max(w, h) > 8 * min(w, h)
        at_edge = x == 0 or y == 0 or x + w == W or y + h == H
        small = max_size is None or max(w, h) <= max_size
        if area >= 30 and not line_like and small and (touch_edge or not at_edge) \
                and (region[y:y + h, x:x + w] > 0).any():
            protect[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i] = 255
    return cv2.dilate(protect, np.ones((5, 5), np.uint8))


def border_dark_mask(img: np.ndarray, bg: tuple[int, int, int], thresh: int, margin: int,
                     keep: np.ndarray | None = None) -> np.ndarray:
    """与图片边缘相连的黑色区域（黑边、三角黑角、撕裂的黑块）的掩码。

    只处理“接触图片边缘”的深色连通域，页面内部的手写、印刷文字不会被触及。
    keep: 印章保护掩码，其中的像素既不作为种子、也不会被生长吞掉。
    """
    H, W = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    seed = edge_dark_seed(gray, bg, thresh)
    free = np.full((H, W), 255, np.uint8) if keep is None else cv2.bitwise_not(keep)
    seed &= free
    if not seed.any():
        return seed
    # 沿渐变的灰色过渡带向外生长（受限于较亮的灰度上限），吃掉黑边的抗锯齿/阴影边缘
    soft = (gray < min(int(np.mean(bg)) - 40, 190)).astype(np.uint8) & (free > 0)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(max(margin, 1) * 3):
        seed = cv2.bitwise_and(cv2.dilate(seed, k3), np.maximum(soft, seed))
    core = cv2.dilate(seed, np.ones((margin * 2 + 1,) * 2, np.uint8))
    # 黑边内侧通常还有一圈偏亮的扫描过曝带或残余暗线，按图像大小再向外多扩一些一并修补
    r = max(margin, int(round(0.022 * min(H, W))))
    grown = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    # 但紧挨黑边的手写/印刷字符不能被扩展区吞掉：保护扩展区里“像字”的墨迹块（细长线条不保护）
    protect = charlike_ink(gray, bg, grown, core)
    return cv2.bitwise_and(cv2.bitwise_or(core, cv2.bitwise_and(grown, cv2.bitwise_not(protect))), free)


def color_noise_clean(img: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """去除大面积、浅淡的彩色噪点（复印机的粉色墨粉斑、扫描仪的彩色竖条纹等），颜色不写死。

    “彩色”按 Lab 色度相对纸张底色计算，所以任何颜色的噪点都能识别；区分噪点与彩色内容靠三点：
      - 浓：色度高的像素（红头、印章的笔画）及其周围不动；
      - 深：深色墨迹（哪怕本身带颜色，如粉色复印件上的字）及紧贴它的边缘不动；
      - 散：剩下的浅色彩色像素按密度连成区域，只有铺满大片、或贯穿页面一大段的区域才算噪点；
        浅淡的印章是一个紧凑的圆，不会被当成噪点。
    噪点像素按色度渐变地替换为纸张底色。
    """
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    chroma = np.hypot(lab[..., 1] - paper_lab[1], lab[..., 2] - paper_lab[2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = float(np.mean(bg))
    side = min(H, W)

    strong = (chroma > COLOR_STRONG).astype(np.uint8)
    r = max(3, int(0.004 * side))
    protect = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    dark = (gray < paper * 0.55).astype(np.uint8)
    protect |= cv2.dilate(dark, np.ones((5, 5), np.uint8))
    cand = ((chroma > COLOR_MIN) & (protect == 0)).astype(np.uint8)
    if cand.mean() < 0.002:
        return img

    # 在缩小图上按密度连成区域
    s = min(1.0, 800 / max(H, W))
    size = (max(1, int(W * s)), max(1, int(H * s)))
    small = cv2.resize(cand.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    strong_s = cv2.resize(strong.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    # 浅色的彩色字（墨粉不足的红头、浅红标题）也是“大片浅色”，但它是成行的、边缘锐利的笔画
    text = find_text(img, bg).text & (cand > 0)
    text_s = cv2.resize(text.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    k = max(3, int(0.02 * min(small.shape)) | 1)
    dens = cv2.blur(small, (k, k))
    region = (dens > 0.08).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    sh, sw = small.shape
    # 印章状的区域：不贴图片边缘、大致成圆、有一定大小。先记下来，后面无论如何都不动
    stamp = np.zeros((sh, sw), np.uint8)
    n, lab_r, st, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        inner = x > 1 and y > 1 and x + w < sw - 1 and y + h < sh - 1
        if inner and max(w, h) <= 2 * min(w, h) and 0.06 * min(sh, sw) <= max(w, h) <= 0.35 * min(sh, sw):
            stamp[lab_r == i] = 255
            log.debug("    彩色区域 (%d,%d) %dx%d 形似印章，保留（缩小图）", x, y, w, h)
    stamp = cv2.dilate(stamp, np.ones((k, k), np.uint8))
    # 条纹多是沿页边的长条，断断续续：顺着横、竖两个方向分别闭运算把断开的片段接起来
    region |= cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((6 * k, 1), np.uint8))
    region |= cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((1, 6 * k), np.uint8))
    n, lab_r, st, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    noise = np.zeros((sh, sw), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        wide = area >= 0.05 * sh * sw or h >= 0.35 * sh or w >= 0.35 * sw
        # 彩色照片、大面积彩色印刷也是“大片彩色”，但其中浓色像素多；噪点几乎全是浅淡的颜色
        sel = lab_r == i
        rich = float(strong_s[sel].sum()) / max(float(small[sel].sum() + strong_s[sel].sum()), 1e-6)
        # 区域里的浅色彩色像素多半是文字笔画：是一片浅色的字，不是噪点
        lettered = float(text_s[sel].sum()) / max(float(small[sel].sum()), 1e-6)
        if wide and rich < 0.1 and lettered >= COLOR_TEXT_FRAC:
            log.debug("    彩色区域 (%d,%d) %dx%d 多为文字笔画（%.2f），视为浅色文字保留（缩小图）", x, y, w, h, lettered)
        elif wide and rich < 0.1:
            noise[sel] = 255
            log.debug("    彩色噪点区域 (%d,%d) %dx%d，浓色占比 %.2f，文字笔画占比 %.2f（缩小图）", x, y, w, h, rich, lettered)
        elif wide:
            log.debug("    彩色区域 (%d,%d) %dx%d 浓色占比 %.2f，视为彩色内容保留（缩小图）", x, y, w, h, rich)
    if not noise.any():
        log.info("  彩色噪点: 未发现")
        return img
    noise = cv2.dilate(noise, np.ones((k, k), np.uint8))
    noise[stamp > 0] = 0
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_NEAREST)

    # 噪点的色相：噪点区域内浅色彩色像素的色相直方图峰值；只清这个色相附近的颜色，
    # 这样噪点区域里别的颜色（如橙红色印章的浅色部分）不受影响
    hue = (np.degrees(np.arctan2(lab[..., 2] - paper_lab[2], lab[..., 1] - paper_lab[1])) + 360) % 360
    sel = (noise > 0) & (cand > 0)
    hist = np.bincount((hue[sel] // 5).astype(np.int32) % 72, minlength=72).astype(np.float64)
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5), "valid")
    peak = float(np.argmax(hist)) * 5 + 2.5
    # 色度很低时色相不稳定，不按色相筛
    near_hue = (np.abs((hue - peak + 180) % 360 - 180) < COLOR_HUE_TOL) | (chroma < 2 * COLOR_MIN)
    log.debug("    噪点色相 %.0f°（Lab a-b 平面）", peak)

    # 色度越高越接近完全替换，低色度渐变，避免留下硬边
    alpha = np.clip((chroma - COLOR_MIN / 3) / COLOR_MIN, 0, 1)
    alpha[(noise == 0) | (protect > 0) | ~near_hue] = 0

    # 墨粉斑里颜色较深的小点会被当成“深色墨迹”保护起来：噪点色相的、很小、周围没有别的墨迹的深色点也清掉
    # （冒号、句号等标点挨着文字，不算孤立）
    dots = (dark > 0) & (chroma > 2 * COLOR_MIN) & near_hue & (noise > 0)
    n_d, lab_d, st_d, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    dot_max, iso = max(4, int(0.005 * side)), max(8, int(0.012 * side))
    # 只看附近有没有“大”墨迹（字、线），成片的小噪点彼此挨着不影响判断
    big = np.zeros(n_d, bool)
    big[1:] = np.maximum(st_d[1:, 2], st_d[1:, 3]) > dot_max
    big_ink = big[lab_d]
    ys, xs = np.nonzero(dots)
    n_dot = 0
    for i in np.unique(lab_d[ys, xs]):
        x, y, w, h, area = st_d[i]
        if big[i]:
            continue
        x0, y0, x1, y1 = max(0, x - iso), max(0, y - iso), min(W, x + w + iso), min(H, y + h + iso)
        if big_ink[y0:y1, x0:x1].any():
            continue
        comp = (lab_d[y0:y1, x0:x1] == i).astype(np.uint8)
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], cv2.dilate(comp, np.ones((5, 5), np.uint8)))
        n_dot += 1
    log.debug("    清除孤立深色噪点 %d 个", n_dot)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]
    log.info("  彩色噪点: 清除 %.2f%% 的面积", 100 * float((alpha > 0.5).mean()))
    out = img.astype(np.float32) * (1 - alpha) + np.float32(bg) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def margin_noise_mask(img: np.ndarray, bg: tuple[int, int, int], noise_ratio: float = NOISE_RATIO,
                      cleaned: np.ndarray | None = None) -> np.ndarray:
    """版心之外的阴影条带与零星噪点的掩码，不含正文。

    noise_ratio: 孤立墨点的最大尺寸占典型字高的比例，越大清除得越多。
    cleaned: 第 1 步清除过的黑边区域（不含边框带，与 img 同坐标），用于判断贴边笔迹是否被黑边截断。
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = float(np.mean(bg))
    ink = (gray < paper * 0.65).astype(np.uint8)
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # 典型字高：取较大墨迹块高度的 75 分位（汉字常被拆成几个部件，取偏上的分位更接近整字高度）
    hs = st2[1:, 3]
    hs = hs[hs > max(10, H // 350)]
    char_h = float(np.percentile(hs, 75)) if hs.size else 0.015 * H

    # 噪点必须明显小于一个字：孤立的整字（如页码“3”、加粗的标题字）不会被误删
    speck = max(6, int(noise_ratio * char_h))

    # 1. 把字符横向合并成行，用墨迹量足够大的连通域确定版心。
    #    墨点不参与：页边成串的墨点离表格框、正文不远时会被横向膨胀并进去，把版心撑到页边
    dot = np.maximum(st2[:, 2], st2[:, 3]) <= speck
    dot[0] = False
    body = ink & ~dot[np.asarray(lab2, np.int32)]
    merge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 25, 3), max(H // 200, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(body, merge), connectivity=8)
    ink_cnt = np.bincount(lab.ravel().astype(np.int32, copy=False), weights=body.ravel(), minlength=n)[1:]
    if ink_cnt.size == 0 or ink_cnt.max() == 0:
        log.info("  去噪: 未发现墨迹，跳过")
        return np.zeros((H, W), np.uint8)
    # 墨迹量至少相当于一个字的才算正文行；不用“相对最大值”的比例，否则一张大照片会把所有文字行都筛掉
    big = np.flatnonzero(ink_cnt >= 0.3 * char_h * char_h) + 1
    boxes = []  # 用墨迹本身的外框（而非膨胀后的）判断，避免细黑线因膨胀被当成正文
    for i in big:
        ys, xs = np.nonzero((lab == i) & (body > 0))
        if ys.size and min(np.ptp(xs), np.ptp(ys)) + 1 > H * 0.008:
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, ink_cnt[i - 1]))
    if not boxes:
        log.info("  去噪: 未找到正文行，无法确定版心，跳过")
        return np.zeros((H, W), np.uint8)
    # 墨迹少、外框又只有一两个字大的块（页角的墨团、污渍也可能这么大）不能单独撑大版心：
    # 先用其余的行定出版心，小块只有离版心足够近时才并入
    def minor(b: tuple) -> bool:
        return b[4] < 2 * char_h * char_h and max(b[2] - b[0], b[3] - b[1]) < 4 * char_h
    major = [b for b in boxes if not minor(b)] or boxes
    x0, y0 = min(b[0] for b in major), min(b[1] for b in major)
    x1, y1 = max(b[2] for b in major), max(b[3] for b in major)
    rest, gap = [b for b in boxes if b not in major], 2 * char_h
    grew = True
    while grew:
        grew = False
        for b in rest[:]:
            if b[0] - x1 <= gap and x0 - b[2] <= gap and b[1] - y1 <= gap and y0 - b[3] <= gap:
                x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
                rest.remove(b)
                grew = True
    for b in rest:
        log.debug("    小块 (%d,%d)-(%d,%d) 远离正文，不计入版心", *b[:4])
    pad_x, pad_y = int(W * 0.02), int(H * 0.015)
    tx0, ty0, tx1, ty1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    log.debug("  去噪: 典型字高 %.0fpx，版心 (%d,%d)-(%d,%d)", char_h, max(tx0, 0), max(ty0, 0),
              min(tx1, W), min(ty1, H))

    outside = np.ones((H, W), np.uint8)
    outside[max(ty0, 0):ty1, max(tx0, 0):tx1] = 0
    kill = np.zeros((H, W), np.uint8)

    # 2. 版心之外的孤立小块墨点：尺寸远小于一个字、接近“团块”形状，并且周围没有字、线这样的大墨迹
    #    （细长的手写笔画如“1”不删；周围有字的是正文，哪怕颜色浅、没被划进版心也不动）
    radius = 0.05 * min(H, W)
    near = int(0.03 * min(H, W))  # 黑边清除后，被截断的笔迹离边缘仍会留有一小段距离
    # 噪点常带一圈浅灰的晕，深色阈值只抓得到核心：按浅色阈值连通，清除时连晕一起去掉
    soft = (gray < paper - 40).astype(np.uint8)
    n3, lab3, st3, cen3 = cv2.connectedComponentsWithStats(soft, connectivity=8)
    lab3 = np.asarray(lab3, np.int32)
    # “大墨迹”：比噪点大的深色块，或接近字大小的浅色块（浅色的字被深色阈值拆成碎块，但浅色阈值下仍是整字）。
    # 只有它们才说明附近是正文；成片的噪点彼此挨着，不能互相证明是文字
    big2 = np.flatnonzero(np.maximum(st2[1:, 2], st2[1:, 3]) > speck) + 1
    # 浅色块要是笔画构成的（填充率低）；噪点的晕是实心的一团，哪怕有字那么大也不算
    big3 = np.flatnonzero((np.maximum(st3[1:, 2], st3[1:, 3]) >= 0.8 * char_h)
                          & (st3[1:, 4] < 0.45 * st3[1:, 2] * st3[1:, 3])) + 1
    # 贴边截断的判断：浅色连通域碰到图片边缘，或碰到第 1 步清除过的黑边区域
    edge_zone = np.zeros((H, W), np.uint8)
    edge_zone[:3], edge_zone[-3:], edge_zone[:, :3], edge_zone[:, -3:] = 1, 1, 1, 1
    if cleaned is not None:
        edge_zone |= (cv2.dilate(cleaned, np.ones((7, 7), np.uint8)) > 0).astype(np.uint8)
    cut = np.zeros(n3, bool)
    cut[np.unique(lab3[(edge_zone > 0) & (soft > 0)])] = True
    cut[0] = False

    # 带颜色的块（印章的碎笔画、彩色墨粉点）只按旧规则判断：成片的印章碎点彼此挨着，不能当孤立噪点
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    chroma = np.hypot(lab_img[..., 1] - paper_lab[1], lab_img[..., 2] - paper_lab[2])

    # 单个小块的色度受 JPEG 噪声影响很大（浅色印章的碎笔画平均色度可能和灰色噪点差不多），
    # 改看周围一片：附近的浅色墨迹里带颜色的占比高，就是印章、彩色笔迹的一部分
    k = 2 * speck + 1
    soft_f = soft.astype(np.float32)
    tint = cv2.boxFilter(soft_f * (chroma > 2 * COLOR_MIN), -1, (k, k), normalize=False)
    tint /= np.maximum(cv2.boxFilter(soft_f, -1, (k, k), normalize=False), 1.0)
    colored3 = np.bincount(lab3.ravel(), weights=tint.ravel(), minlength=n3) / np.maximum(st3[:, 4], 1) > 0.3
    colored3[0] = False

    def colored(sel: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        return bool(colored3[lab3[y:y + h, x:x + w][sel]].any())

    def out_c(c: np.ndarray) -> np.ndarray:  # 中心落在版心之外
        return outside[np.minimum(c[:, 1], H - 1).astype(int), np.minimum(c[:, 0], W - 1).astype(int)] > 0

    # 版心里的字离版心外的墨点至少隔着版心的留白，不能说明墨点是正文（如表格框内紧挨边框的字）；
    # 只看同在版心外的大墨迹（页边的手写字、浅色的字行）
    big2o, big3o = big2[out_c(cen2[big2])], big3[out_c(cen3[big3])]

    def no_big_near(cx: float, cy: float, own2: int, own3: np.ndarray) -> bool:
        # 附近没有字、线这样的大墨迹；不算自己，也不算包住自己的浅色晕
        b2, b3 = big2o[big2o != own2], big3o[~np.isin(big3o, own3)]
        xy = np.r_[cen2[b2], cen3[b3]]
        return not (np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < radius).any()

    # 浅色阈值下的“团块”：填充率高、没被图片边缘截断、不比墨点大多少。
    # 装订孔轮廓、浅色骑缝残迹、浅色的字是细线笔画，填充率低
    def lump(j: int | np.integer) -> bool:
        return bool(st3[j, 4] >= 0.3 * st3[j, 2] * st3[j, 3]) and not cut[j] \
            and max(st3[j, 2], st3[j, 3]) <= 1.2 * speck

    # 笔画构成的字（字大小、填充率低，不含长线、表格框、实心墨团）：紧挨着它、落在同一行中段的
    # 短横或小点是标点（如页码“- 2 -”两侧的短横）
    sz2 = np.maximum(st2[big2, 2], st2[big2, 3])
    char2 = big2[(sz2 <= 2 * char_h) & (st2[big2, 3] >= 0.4 * char_h)
                 & (st2[big2, 4] < 0.45 * st2[big2, 2] * st2[big2, 3])]

    def beside_char(x: int, y: int, w: int, h: int) -> bool:
        dash = w >= 2 * h and h <= 0.25 * char_h
        dot = max(w, h) <= 0.25 * char_h
        if not (dash or dot):
            return False
        bx, by, bw, bh = (st2[char2, k] for k in range(4))
        cy = y + h / 2
        mid = (by + 0.2 * bh <= cy) & (cy <= by + 0.8 * bh)
        gap = np.maximum(bx - (x + w), x - (bx + bw))  # 水平间距，重叠时为负
        return bool((mid & (gap >= 0) & (gap <= char_h)).any())

    blob = np.zeros((H, W), np.uint8)  # 较大的墨团，外面还会再清一圈淡晕

    def erase(sel: np.ndarray, x: int, y: int, w: int, h: int, halo: bool, lump_only: bool = False) -> None:
        # halo: 连同包住它的浅色晕一起清。晕若大得像一片字（浅色正文压着墨点）、或带颜色（印章笔画、彩色墨粉团），
        # 或者深色核心只有一两个像素（多是浅色印章、浅色图形里的深色像素），只清深色部分；
        # 但晕本身是个小团块时，核心再小也是墨点。lump_only: 只清团块状的晕（页边手写字的碎笔画，
        # 其浅色连通域是整个字，不能连带清掉）
        kill[y:y + h, x:x + w][sel] = 255
        if not halo:
            return
        for j in np.unique(lab3[y:y + h, x:x + w][sel]):
            if (int(sel.sum()) < 4 or lump_only) and not lump(j):
                continue
            if j and max(st3[j, 2], st3[j, 3]) <= 3 * speck and not colored3[j]:
                jx, jy, jw, jh = st3[j, :4]
                comp = lab3[jy:jy + jh, jx:jx + jw] == j
                kill[jy:jy + jh, jx:jx + jw][comp] = 255
                if max(jw, jh) >= max(4, 0.15 * char_h):
                    blob[jy:jy + jh, jx:jx + jw][comp] = 255

    n_edge = n_speck = 0
    seen = np.zeros(n3, bool)  # 含深色核心的浅色连通域，由核心来判
    seen[np.unique(lab3[ink > 0])] = True
    for i in range(1, n2):
        x, y, w, h, area = st2[i]
        cx, cy = cen2[i]
        if not outside[int(min(cy, H - 1)), int(min(cx, W - 1))]:
            continue
        sel = lab2[y:y + h, x:x + w] == i
        solid = area >= 0.45 * w * h
        # 版心之外、紧贴（或被截断于）图片边缘的笔迹：多半是相邻页透过来的字或扫描残留，直接去掉。
        # 但没被截断、字大小、笔画构成的块是页边完整的手写字（如页码、档案编号“4”），保留。
        # 没被截断的小块是页边的墨点，常常只有一小撮深色核心、外面是一大团灰晕，连晕一起清
        if x <= near or y <= near or x + w >= W - near or y + h >= H - near:
            truncated = bool(cut[lab3[y:y + h, x:x + w][sel]].any())
            if truncated or max(w, h) < 0.6 * char_h or solid:
                erase(sel, x, y, w, h, halo=not truncated, lump_only=True)
                n_edge += 1
                log.debug("    清除贴边笔迹 (%d,%d) %dx%d", x, y, w, h)
            else:
                log.debug("    保留页边手写字 (%d,%d) %dx%d", x, y, w, h)
            continue
        # 接近字大小的块只有“实心墨团”才算噪点（笔画构成的字填充率低）
        if max(w, h) > (2 * speck if solid else speck) or max(w, h) > 3 * min(w, h):
            continue
        # 周围墨迹块不超过两个（旧规则），或者附近只有噪点、没有大墨迹（灰色的成片噪点）
        few = int((np.hypot(cen2[1:, 0] - cx, cen2[1:, 1] - cy) < radius).sum()) - 1 <= 2
        if not few and (colored(sel, x, y, w, h) or not no_big_near(cx, cy, i, np.unique(lab3[y:y + h, x:x + w][sel]))):
            continue
        if beside_char(x, y, w, h):
            log.debug("    保留字旁的标点 (%d,%d) %dx%d", x, y, w, h)
            continue
        erase(sel, x, y, w, h, halo=True)
        n_speck += 1
        log.debug("    清除孤立墨点 (%d,%d) %dx%d", x, y, w, h)

    # 没有深色核心的浅灰噪点：同样要求小、成团、附近没有大墨迹。带颜色的（印章碎笔画）不管。
    # 页边一圈只清团块：装订孔、浅色的骑缝残迹常在那里，只露出零碎的浅色笔画
    n_faint = 0
    for j in np.flatnonzero(~seen[1:]) + 1:
        x, y, w, h, area = st3[j]
        cx, cy = cen3[j]
        if not outside[int(min(cy, H - 1)), int(min(cx, W - 1))] or area < 10:
            continue
        at_edge = x <= near or y <= near or x + w >= W - near or y + h >= H - near
        if at_edge and not lump(j):
            continue
        sel = lab3[y:y + h, x:x + w] == j
        # 团块的浅色阈值连通域比深色核心大一圈，尺寸上限放宽一些
        lim = 1.2 * speck if lump(j) else speck
        if max(w, h) > lim or max(w, h) > 3 * min(w, h) or colored3[j] or not no_big_near(cx, cy, 0, np.array([j])):
            continue
        # 紧挨着还有没清掉的浅色笔画，说明它是某个浅色图形（装订孔轮廓、浅色字）的一部分；
        # 成片的小团块彼此挨着不算
        g = 2 * speck
        wx0, wy0, wx1, wy1 = max(0, x - g), max(0, y - g), min(W, x + w + g), min(H, y + h + g)
        others = lab3[wy0:wy1, wx0:wx1][(soft[wy0:wy1, wx0:wx1] > 0) & (kill[wy0:wy1, wx0:wx1] == 0)]
        others = np.unique(others[others != j])
        if any(st3[o, 4] >= 10 and not lump(o) for o in others):
            continue
        kill[y:y + h, x:x + w][sel] = 255
        n_faint += 1
        log.debug("    清除浅色墨点 (%d,%d) %dx%d", x, y, w, h)

    # 版心内的散点（表格空白格里成片的墨粉点、印章透印的碎点）：比墨点上限小、半个字高内没有任何大墨迹
    # （字、笔画、表格线），也不带颜色。字旁的标点、小数点离字很近，紧贴表格线的点也不动
    r_in = max(3, int(0.5 * char_h))
    # 表格线、下划线这样的长直线不说明附近有字，不保护紧挨着它的散点
    # 长度要远超一行里相邻几个字的横笔连起来的长度，否则整行字会被当成直线挖掉
    Ln = max(int(5 * char_h), W // 20, H // 20)
    faint = (gray < paper - 15).astype(np.uint8)  # 按浅色阈值找，颜色偏浅的一段表格线也算
    # 先补上线里的小断口（印刷、扫描造成的白点），断续的表格线才能被整条找出来
    straight = cv2.morphologyEx(cv2.morphologyEx(faint, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)),
                                cv2.MORPH_OPEN, np.ones((1, Ln), np.uint8)) \
        | cv2.morphologyEx(cv2.morphologyEx(faint, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8)),
                           cv2.MORPH_OPEN, np.ones((Ln, 1), np.uint8))
    straight &= cv2.dilate(faint, np.ones((3, 3), np.uint8))  # 只留真有墨迹的地方，不把补出来的断口当线
    line_px = cv2.dilate(straight, np.ones((7, 7), np.uint8)) * 255  # 清散点时不许碰表格线本身
    straight = cv2.dilate(straight, np.ones((5, 5), np.uint8))
    inner = np.zeros((H, W), np.uint8)
    # 先去掉长直线再分连通域：粘在表格线上的碎点也能单独拆出来，不再算作表格这个大墨迹的一部分
    n8, lab8, st8, cen8 = cv2.connectedComponentsWithStats(ink & (straight == 0), connectivity=8)
    big8 = np.maximum(st8[:, 2], st8[:, 3]) > speck
    big8[0] = False
    big_px = big8[lab8].astype(np.uint8) * 255
    big_near = cv2.dilate(big_px,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1,) * 2))
    # 墨粉不足的字本身就断成一串小点，单看大小和邻居分不出来：正文行（按行结构识别）里一概不动
    band = cv2.dilate(find_text(img, bg).band.astype(np.uint8), np.ones((r_in, r_in), np.uint8))
    big_near |= band
    # 行外的小号字（表格格子里的标签、页末的短行）同样会碎成小块：相距很近的碎块连成一团后，
    # 字的团有小半个字以上大、团内墨迹密；散点要么单独一颗，要么连成的团很稀疏
    kc = max(3, int(0.1 * char_h)) | 1
    clus = cv2.dilate((ink & (straight == 0)).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    n9, lab9, st9, _ = cv2.connectedComponentsWithStats(clus, connectivity=8)
    fill9 = np.bincount(lab9[(ink > 0) & (straight == 0)], minlength=n9) / np.maximum(st9[:, 4], 1)
    # 印得很淡的小字团内墨迹也稀，但碎块里总有细长的笔画；墨粉点是一颗颗圆团
    long8 = np.maximum(st8[:, 2], st8[:, 3])
    stroke8 = (long8 >= 2.5 * np.minimum(st8[:, 2], st8[:, 3])) & (long8 >= 0.15 * char_h)
    stroke8[0] = False
    has_stroke = np.zeros(n9, bool)
    has_stroke[np.unique(lab9[stroke8[lab8]])] = True
    charlike = (np.maximum(st9[:, 2], st9[:, 3]) >= 0.2 * char_h) & ((fill9 >= 0.25) | (fill9 >= 0.15) & has_stroke)
    charlike[0] = False
    # 和大墨迹一样保护半个字高：字旁的冒号、顿号离字稍远，也不能当散点
    big_near |= cv2.dilate(charlike[lab9].astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1,) * 2))
    # 散点周围还有一层只比纸色暗十几个灰度的雾点。按这个浅色阈值连通：印刷的字哪怕深色部分断成碎块，
    # 浅色下笔画仍连成字大小的一块；墨粉散点在浅色下也还是一颗颗分开的小点
    haze = faint & (straight == 0)
    n7, lab7, st7, cen7 = cv2.connectedComponentsWithStats(haze, connectivity=8)
    lab7 = np.asarray(lab7, np.int32)
    small7 = np.maximum(st7[:, 2], st7[:, 3]) <= speck
    small7[0] = False
    touch = np.zeros(n7, bool)  # 碰到大墨迹附近、或带颜色的不动
    touch[np.unique(lab7[(big_near > 0) | (chroma > 2 * COLOR_MIN)])] = True
    in_body = ~out_c(cen7)
    fog_ok = small7 & ~touch & in_body
    n_inner = 0
    for i in range(1, n8):
        x, y, w, h, area = st8[i]
        cx, cy = cen8[i]
        if outside[int(min(cy, H - 1)), int(min(cx, W - 1))] or max(w, h) > speck \
                or (max(w, h) > 4 and max(w, h) > 3 * min(w, h)):
            continue
        sel = lab8[y:y + h, x:x + w] == i
        x0_, y0_, x1_, y1_ = max(0, x - 2), max(0, y - 2), min(W, x + w + 2), min(H, y + h + 2)
        if big_near[y:y + h, x:x + w][sel].any() or colored(sel, x, y, w, h) \
                or not small7[lab7[y:y + h, x:x + w][sel]].all() \
                or straight[y0_:y1_, x0_:x1_].any():  # 紧贴表格线的可能是线抖开的一截
            continue
        inner[y:y + h, x:x + w][sel] = 255
        n_inner += 1
    # 雾点留着会被修补抹成灰色拖影：同样远离大墨迹的浅灰小块一并清掉；
    # 但块里有没判成散点的深色笔画，就是字的一部分，不是雾
    fog_ok[np.unique(lab7[(ink > 0) & (inner == 0)])] = False
    inner[fog_ok[lab7]] = 255
    log.info("  去噪: 清除版心内散点 %d 处、浅灰雾点 %d 处", n_inner, int(fog_ok.sum()))

    # 页边的直线角标、裁切线残段（多是扫描仪留下的灰色细线）：浅色阈值下整块几乎全由横、竖直线构成，
    # 而且是一横一竖拐角相接的角标。单独的竖线可能是手写的“1”或装订孔方框断开的一边，
    # 三面、四面都有边的是装订孔的方框，都不动
    L = max(8, int(0.3 * char_h))
    hl = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    vl = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    n_line = 0
    for j in range(1, n3):
        x, y, w, h, area = st3[j]
        cx, cy = cen3[j]
        if not outside[int(min(cy, H - 1)), int(min(cx, W - 1))] or max(w, h) < L or max(w, h) > 4 * char_h:
            continue
        if not (x <= near or y <= near or x + w >= W - near or y + h >= H - near) or colored3[j]:
            continue
        sel = lab3[y:y + h, x:x + w] == j
        nh, nv = int(hl[y:y + h, x:x + w][sel].sum()), int(vl[y:y + h, x:x + w][sel].sum())
        if (nh + nv) < 0.6 * area:
            continue
        # 外框四条边上各有多少墨迹：被线覆盖过半的算一条边
        c = sel.astype(np.uint8)
        t = 3
        sides = [c[:t].any(0).mean(), c[-t:].any(0).mean(), c[:, :t].any(1).mean(), c[:, -t:].any(1).mean()]
        n_sides = sum(v > 0.5 for v in sides)
        corner = nh >= L and nv >= L and n_sides == 2 and min(w, h) >= L / 2
        if not corner:
            continue
        kill[y:y + h, x:x + w][sel] = 255
        n_line += 1
        log.debug("    清除页边直线角标 (%d,%d) %dx%d", x, y, w, h)

    log.info("  去噪: 清除贴边笔迹 %d 处、孤立墨点 %d 处、浅色墨点 %d 处、直线角标 %d 处（墨点尺寸上限 %dpx = 字高 %.0fpx × %.2f）",
             n_edge, n_speck, n_faint, n_line, speck, char_h, noise_ratio)
    kill = cv2.dilate(kill, np.ones((7, 7), np.uint8))
    # 外扩时不许碰表格线（连同两侧）和字、笔画这样的大墨迹
    kill |= cv2.dilate(inner, np.ones((7, 7), np.uint8)) & cv2.bitwise_not(line_px | big_px)
    # 大一点的墨团外面还有一圈更淡的晕，只清到浅色阈值会留下一个灰影：连着它的淡色小块一并清掉
    # （一两个像素的小点不扩，它们常是浅色印章、浅色字里的深色像素）
    if blob.any():
        halo = (gray < paper - 20).astype(np.uint8)
        n4, lab4, st4, _ = cv2.connectedComponentsWithStats(halo, connectivity=8)
        hit = np.zeros(n4, bool)
        hit[np.unique(lab4[(blob > 0) & (halo > 0)])] = True
        hit[0] = False
        hit &= np.maximum(st4[:, 2], st4[:, 3]) <= 4 * speck
        # 晕连着的若是带颜色的块（彩色墨粉团、印章碎笔画），不扩过去
        hit &= ~(np.bincount(lab4.ravel(), weights=tint.ravel(), minlength=n4) / np.maximum(st4[:, 4], 1) > 0.3)
        kill |= cv2.dilate(hit[lab4].astype(np.uint8) * 255, np.ones((5, 5), np.uint8))

    # 页边的淡色痕迹：背面透过来的字、纸张折角和折痕的阴影。它们只比纸色暗几个灰度，上面的阈值都抓不到。
    # 按局部纸色（大窗口中值，不受扫描明暗不均影响）找出略暗的区域（滞后阈值：明显偏暗的连着的稍暗部分），
    # 只在远离正文行、附近也没有真正墨迹（页边的手写字和它的淡晕）、不带颜色（浅色印章）的地方清除
    text = np.zeros((H, W), np.uint8)
    for b in boxes:
        if b not in rest:
            text[max(b[1] - pad_y, 0):b[3] + pad_y, max(b[0] - pad_x, 0):b[2] + pad_x] = 1
    sg = cv2.resize(gray, (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA)
    local = cv2.resize(cv2.medianBlur(sg, 31), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    diff = local - cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    # 只在版心外、以及页边一条（折角常从页角斜着伸进版心的外框）里清，正文行之间的淡色不管
    rim = np.zeros((H, W), np.uint8)
    e = int(0.1 * min(H, W))
    rim[:e], rim[-e:], rim[:, :e], rim[:, -e:] = 1, 1, 1, 1
    weak = ((diff > 3.5) & (text == 0) & ((outside > 0) | (rim > 0))).astype(np.uint8)
    n5, lab5 = cv2.connectedComponents(weak, connectivity=8)
    hit = np.zeros(n5, bool)
    hit[np.unique(lab5[(diff > 7) & (weak > 0)])] = True
    hit[0] = False
    # 真正的墨迹：深色像素，以及不比墨点小的浅色笔画（装订孔方框、浅色手写字）；没清掉的零碎浅色小点不算
    real = (gray < paper - 50) | (np.maximum(st3[:, 2], st3[:, 3]) >= speck)[lab3] & (soft > 0)
    # 灰色的阴影、污渍（如页顶的手印、压痕）中心也可能比纸暗 50 以上，但它不是笔画：成片的略暗区域里
    # 有远比笔画厚的地方。这样的整片（连同陡峭的边沿）里中等深度的像素不算墨迹；真正的黑墨迹不管在哪都算
    shade = (gray < paper - 25).astype(np.uint8)
    n6, lab6 = cv2.connectedComponents(shade, connectivity=8)
    lab6 = np.asarray(lab6, np.int32)
    dt = cv2.distanceTransform(shade, cv2.DIST_L2, 3)
    thick = np.zeros(n6, bool)
    thick[np.unique(lab6[dt > max(4.0, 0.3 * char_h)])] = True
    thick[0] = False
    real = ((gray < paper - 120) | real & ~thick[lab6]) & (kill == 0)
    real = real.astype(np.uint8)
    r = max(3, int(0.5 * char_h))
    guard = cv2.dilate(real, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    guard = cv2.bitwise_or(guard, cv2.dilate((chroma > 2 * COLOR_MIN).astype(np.uint8), np.ones((9, 9), np.uint8)))
    wash = hit[lab5] & (guard == 0)
    log.info("  去噪: 清除页边淡色痕迹 %.2f%% 的面积", 100 * float(wash.mean()))
    kill = cv2.bitwise_or(kill, cv2.dilate(wash.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)))
    return kill


def edge_band_mask(img: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """扫描仪在页边留下的“边框带”（过曝白带、细暗线、亮度台阶）的掩码。

    四条边各取中间 60% 长度的逐行/列中位数，从边缘向内找出与纸张底色持续存在差异的区间，
    该区间整体交给修补，这样才不会留下一条分界线。
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((H, W), np.uint8)

    def band(profile: np.ndarray, limit: int) -> int:
        paper = float(np.median(profile[limit:limit * 2 + 1]))
        tol, gap_max = max(3.0, 0.015 * paper), 3
        last, gap = 0, 0
        for k in range(limit):
            if abs(float(profile[k]) - paper) > tol:
                last, gap = k + 1, 0
            else:
                gap += 1
                if gap > gap_max:
                    break
        return last

    lim_h, lim_w = max(8, int(0.04 * H)), max(8, int(0.04 * W))
    mid_r, mid_c = slice(int(H * 0.2), int(H * 0.8)), slice(int(W * 0.2), int(W * 0.8))
    top = band(np.median(gray[:lim_h * 2 + 1, mid_c], axis=1), lim_h)
    bottom = band(np.median(gray[-lim_h * 2 - 1:, mid_c], axis=1)[::-1], lim_h)
    left = band(np.median(gray[mid_r, :lim_w * 2 + 1], axis=0), lim_w)
    right = band(np.median(gray[mid_r, -lim_w * 2 - 1:], axis=0)[::-1], lim_w)
    mask[:top], mask[H - bottom:], mask[:, :left], mask[:, W - right:] = 255, 255, 255, 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    # 边框带常是扫描的亮度渐变，可能有几十像素宽，压在页边的手写字（如档案编号）不能一起抹掉；
    # 细暗线、长横线、贴边被截断的残迹不保护
    protect = charlike_ink(gray, bg, mask, touch_edge=False, max_size=int(0.08 * min(H, W)))
    return cv2.bitwise_and(mask, cv2.bitwise_not(protect))


def inpaint(img: np.ndarray, mask: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """用周围纸张纹理修补掩码区域，并在边界羽化过渡，避免出现色块和明显分界。

    keep: 印章保护掩码，其中的像素原样保留（羽化也不会渗入）。
    """
    if keep is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(keep))
    if not mask.any():
        return img
    H, W = img.shape[:2]
    scale = max(1.0, min(H, W) / 600)  # 在缩小图上修补，大面积也快
    # 掩码附近的字迹也当作“未知”参与修补计算，否则会把笔画的深色抹进修补区形成灰色拖影；
    # 最终只替换原掩码内的像素，这些字迹本身保持原样
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (gray < float(np.median(gray)) * 0.65).astype(np.uint8)
    reach = max(3, int(0.03 * min(H, W)))
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2))
    work = cv2.bitwise_or(mask, cv2.dilate(cv2.bitwise_and(ink, near), np.ones((5, 5), np.uint8)))
    if keep is not None:  # 印章也不能当修补的取样来源，否则红色会被抹进修补区形成粉色光晕
        work = cv2.bitwise_or(work, cv2.bitwise_and(keep, near))
    small = cv2.resize(img, (int(W / scale), int(H / scale)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(work, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    fixed = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    fixed = cv2.resize(fixed, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # 修补结果本身较平滑，叠加原图的高频纸纹噪声的缺失在此可忽略；按羽化后的掩码混合
    sigma = max(1.0, 0.002 * min(H, W))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)[..., None]
    alpha = np.maximum(alpha, (mask > 0)[..., None].astype(np.float32))
    alpha[(ink > 0) & (mask == 0)] = 0  # 羽化不渗入掩码外的字迹、表格线（修补结果里它们被当作未知抹白了）
    if keep is not None:
        alpha[keep > 0] = 0
    return np.clip(img * (1 - alpha) + fixed * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 文字增强

class TextInfo(NamedTuple):
    text: np.ndarray   # 正文行里的笔画（bool）
    wide: np.ndarray   # 其中属于够宽的正文行的笔画（bool）
    band: np.ndarray   # 正文行所在的区域（bool），笔画之间的空白也在内
    ext_dark: np.ndarray  # 行延伸出来的部分里，源行笔画的暗度（黑帽值 95 分位）；其余为 0
    char_h: float      # 字高
    bh: np.ndarray     # 笔画尺度的暗度（黑帽值）
    ds: np.ndarray     # 平滑后的黑帽值
    lo: float          # 笔画暗度下限
    n_lines: int       # 正文行数


def find_text(img: np.ndarray, bg: tuple[int, int, int]) -> TextInfo:
    """找出白纸上正文行里的文字笔画（包括淡到只比纸色暗几个灰度的字）。

      - 暗度按笔画尺度算（黑帽变换）：比笔画宽得多的阴影、污渍、折痕、亮度渐变不计；
      - 笔画按行组织：锐利边缘在一个字大小窗口里的密度连成行段，只有暗度够、边缘锐利（字是印出来的，
        边缘清楚；污渍、透字、扫描阴影边缘发虚）、笔画不粗的段才算正文；
      - 已确认的行沿行方向延伸到整个版心宽度（整片墨粉不足的淡字也算进来）；
      - 底色偏暗（页边阴影、色块）的地方、与图片边缘相连的墨迹不算。
    行外的零星墨点、成片污渍一概不算文字。
    """
    H, W = img.shape[:2]
    side = min(H, W)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(9, int(ENHANCE_STROKE * side) | 1)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    ds = cv2.GaussianBlur(bh, (0, 0), max(1.0, 0.0008 * side))  # 扫成断点的笔画连成线
    # 大部分像素是纸：以高分位作为纸面噪声水平，笔画至少要高出它
    lo = max(ENHANCE_MIN_DARK, 2.5 * float(np.percentile(ds, 85)))
    weak = (ds > lo).astype(np.uint8)
    # 正文行：锐利边缘（印刷字的笔画边缘清楚，污渍、透字、扫描阴影边缘发虚）在一个字大小的窗口里的密度。
    # 淡字的笔画常被扫成断点，按单个连通域找会漏掉整段，按密度找则整行连成一片
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edge = ((grad > ENHANCE_EDGE) & (weak > 0)).astype(np.float32)
    kx, ky = max(3, int(0.03 * side)), max(3, int(0.01 * side))
    seg = (cv2.blur(edge, (kx, ky)) > ENHANCE_EDGE_DENSITY).astype(np.uint8)
    ns, labs, ss, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
    labs = np.asarray(labs, np.int32)
    # 字高：够长的行段高度的中位数（减去窗口带来的外扩）
    tall = ss[1:, 3][(ss[1:, 2] >= 2 * ss[1:, 3]) & (ss[1:, 2] >= 0.08 * side)]
    char_h = max(float(np.median(tall)) - ky, 8.0) if tall.size else 0.02 * H

    # 逐段检查是否像文字：暗度够、边缘锐利、笔画不粗（成片的污渍、阴影最厚处远超笔画宽度）
    ok = np.zeros(ns, bool)
    for i in range(1, ns):
        x, y, w, h, _ = ss[i]
        sel = (labs[y:y + h, x:x + w] == i) & (weak[y:y + h, x:x + w] > 0)
        if int(sel.sum()) < 0.1 * char_h * char_h:
            continue
        if h > 6 * char_h:  # 标题字可能有正文的两三倍高，再高就是成片的东西
            log.debug("    文字行: 区域 (%d,%d) %dx%d 太高，不算", x, y, w, h)
            continue
        sharp = float(np.percentile(grad[y:y + h, x:x + w][sel], 90))
        dark = float(np.percentile(bh[y:y + h, x:x + w][sel], 95))
        dt = cv2.distanceTransform(np.pad(sel, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        thick = float(np.percentile(dt[sel], 95))
        if sharp >= ENHANCE_SHARP and dark >= ENHANCE_LINE_DARK and thick <= 0.3 * char_h:
            ok[i] = True
        else:
            log.debug("    文字行: 区域 (%d,%d) %dx%d 不像文字（锐度 %.0f，暗度 %.0f，粗细 %.1f）",
                      x, y, w, h, sharp, dark, thick)
    # 合格的段沿行方向接起来：同一行里极淡的几个字夹在中间也算正文
    line = cv2.morphologyEx(ok[labs].astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((1, max(3, int(0.06 * side))), np.uint8))
    nl, labl, sl, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    labl = np.asarray(labl, np.int32)
    good = np.zeros(nl, bool)
    good[1:] = (sl[1:, 2] >= 1.5 * char_h) | (sl[1:, 4] >= 2 * char_h * char_h)  # 至少一两个字
    # 墨粉不足常是整页中间一大片淡到几乎看不见（边缘梯度都不够），一行里只剩两头被认出来：
    # 已确认的行沿行方向延伸到整个版心宽度，行内这一段的淡笔画也算正文
    wide = [i for i in np.flatnonzero(good) if sl[i, 2] >= 0.2 * W]
    # 延伸出来的部分记下源行笔画的暗度：延伸本是为墨粉不足的淡行准备的，本来就清楚的行延伸出去的地方
    # （行尾的空白、背面透字）不该加深
    ext_dark = np.zeros((H, W), np.float32)
    if wide:
        bx0 = int(min(sl[i, 0] for i in wide))
        bx1 = int(max(sl[i, 0] + sl[i, 2] for i in wide))
        # 版心左右两侧之外的短“行”是页边的装订孔、记号、扫描灰带上的碎纹，不是正文
        cx = sl[:, 0] + sl[:, 2] / 2
        good &= (sl[:, 2] >= 0.2 * W) | ((cx >= bx0 - char_h) & (cx <= bx1 + char_h))
    band = good[labl]
    if wide:
        for i in np.flatnonzero(good):
            x, y, w, h, _ = sl[i]
            # 只延伸本身就有几个字长的行：页边的短小记号、角标延伸出去会扫过背面透字、空白处
            if h <= 2 * char_h and w >= 4 * char_h:
                band[y:y + h, min(bx0, x):max(bx1, x + w)] = True
                own = (labl[y:y + h, x:x + w] == i) & (weak[y:y + h, x:x + w] > 0)
                d = float(np.percentile(bh[y:y + h, x:x + w][own], 95)) if own.any() else 0.0
                sub = ext_dark[y:y + h, min(bx0, x):max(bx1, x + w)]
                np.maximum(sub, d, out=sub)
        # 延伸出来的部分没经过逐段检查，可能压到页角的污渍、阴影：其中成片的厚块（远比笔画粗）不算
        dt = cv2.distanceTransform(weak, cv2.DIST_L2, 3)
        r = max(3, int(0.3 * char_h))
        blob = cv2.dilate((dt > 0.15 * char_h).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        band &= good[labl] | (blob == 0)
    # 紧贴图片边缘的一圈是扫描边缘、相邻页残迹，不是正文
    e = max(3, int(0.015 * side))
    band[:e], band[-e:], band[:, :e], band[:, -e:] = False, False, False, False
    # 只认白纸上的字：底色本身偏暗的地方（页边阴影、色块里的反白字、成片的污渍）不动
    paper = float(np.mean(bg))
    # （底色取几个字大小窗口里的中值：字只占一小部分，中值就是纸色；不能用闭运算，色块里的反白小字会把它撑亮）
    sm = cv2.resize(gray, (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA)
    base = cv2.resize(cv2.medianBlur(sm, 15), (W, H), interpolation=cv2.INTER_LINEAR)
    band &= base >= paper - ENHANCE_BASE_TOL
    text = ((weak > 0) & band).astype(np.uint8)
    # 和图片边缘连着的（页边阴影、扫描边缘的灰带）不算
    labt = np.asarray(cv2.connectedComponents(text, connectivity=8)[1], np.int32)
    edge_lab = np.unique(np.r_[labt[0], labt[-1], labt[:, 0], labt[:, -1],
                               labt[e], labt[-e - 1], labt[:, e], labt[:, -e - 1]])
    text = (text > 0) & ~np.isin(labt, edge_lab[edge_lab > 0])
    # 够宽的正文行（至少占版面五分之一宽）本身的笔画：零碎的短“行”可能是条纹、污渍的碎块
    wide_line = np.isin(labl, [i for i in np.flatnonzero(good) if sl[i, 2] >= 0.2 * W])
    ext_dark[good[labl]] = 0  # 行本身按局部判断，只有延伸出来的部分按源行
    return TextInfo(text, text & wide_line, band, ext_dark, char_h, bh, ds, lo, int(good.sum()))


def repair_stroke_gaps(ink: np.ndarray, signal: np.ndarray, valid: np.ndarray,
                       char_h: float, noise: float) -> np.ndarray:
    """用 Zhang 骨架端点及双向切线匹配修补短断笔，保留原有笔画宽度。

    skeletonize 来自 scikit-image；只有两端互相朝向、间距短、路径无第三条
    笔画的候选才连接。较长断口还须有浅墨迹支持，不按字义猜测缺失内容。
    """
    core = (ink >= 0.25) & valid
    if not core.any():
        return ink
    skeleton = skeletonize(core, method="zhang").astype(np.uint8)
    degree = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    ys, xs = np.nonzero((skeleton > 0) & (degree == 1))
    if len(xs) < 2:
        return ink
    points = np.column_stack((xs, ys)).astype(np.float64)
    max_gap = float(np.clip(0.2 * char_h, 3.0, 12.0))
    radius = max(3, int(round(min(5, 0.1 * char_h))))
    tangents = np.zeros_like(points)
    h, w = ink.shape
    for i, (x, y) in enumerate(zip(xs, ys)):
        x0, x1, y0, y1 = max(0, x-radius), min(w, x+radius+1), max(0, y-radius), min(h, y+radius+1)
        patch = skeleton[y0:y1, x0:x1]
        # 只沿本端点所在分支追踪，邻近的平行笔画不参与方向估计。
        _, labels = cv2.connectedComponents(patch, connectivity=8)
        yy, xx = np.nonzero(labels == labels[y-y0, x-x0])
        offsets = np.column_stack((xx+x0-x, yy+y0-y))
        far = np.linalg.norm(offsets, axis=1) >= 2
        if far.any():
            direction = -offsets[far].mean(axis=0)
            norm = np.linalg.norm(direction)
            if norm > 0.5:
                tangents[i] = direction / norm
    candidates = []
    tree = cKDTree(points)
    for i, j in sorted(tree.query_pairs(max_gap)):
        delta = points[j] - points[i]
        distance = float(np.linalg.norm(delta))
        if distance < 2:
            continue
        direction = delta / distance
        alignment = min(float(tangents[i] @ direction), float(-tangents[j] @ direction))
        if alignment < 0.8:
            continue
        # 在像素中心采样直线；拒绝保护区、越界及穿过其他笔画的连接。
        samples = np.rint(np.linspace(points[i], points[j], max(3, int(distance*2)+1))).astype(int)
        px, py = samples[:, 0], samples[:, 1]
        if not valid[py, px].all():
            continue
        occupied = core[py, px]
        empty = np.flatnonzero(~occupied)
        if not len(empty):
            continue
        first, last = int(empty[0]), int(empty[-1])
        if occupied[first:last+1].any():
            continue
        # 粗笔画的骨架端点会向内退缩；用真正空白段的长度判断缺口。
        gap_span = int(np.max(np.abs(samples[last]-samples[first]))) + 1
        gap_signal = signal[py[first:last+1], px[first:last+1]]
        evidence = float(np.mean(gap_signal > max(1.0, 0.35*noise)))
        if gap_span > 3 and evidence < 0.35:
            continue
        candidates.append((distance / alignment, i, j))
    out = ink.copy()
    used = set()
    depth = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    repaired = 0
    for _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        x, y = xs[i], ys[i]
        qx, qy = xs[j], ys[j]
        pad = 3
        x0, x1 = max(0, min(x,qx)-pad), min(w, max(x,qx)+pad+1)
        y0, y1 = max(0, min(y,qy)-pad), min(h, max(y,qy)+pad+1)
        line = np.zeros((y1-y0, x1-x0), np.uint8)
        width = int(np.clip(round(2*min(depth[y,x], depth[qy,qx])-1), 1, 3))
        cv2.line(line, (int(x-x0),int(y-y0)), (int(qx-x0),int(qy-y0)), 255, width, cv2.LINE_AA)
        strength = min(0.85, float(ink[y,x]), float(ink[qy,qx]))
        patch = out[y0:y1,x0:x1]
        bridge = line.astype(np.float32)/255*strength
        bridge[~valid[y0:y1,x0:x1]] = 0
        original = ink[y0:y1, x0:x1]
        if ((patch > 0.25) & (original < 0.25) & (bridge > 0.1)).any():
            continue  # 不允许后续补线穿过已经新增的连接
        np.maximum(patch, bridge, out=patch)
        used.update((i,j))
        repaired += 1
    log.info("  断笔修复: 连接 %d 处（最大端点间距 %.1fpx）", repaired, max_gap)
    return out


def directional_strokes(bh: np.ndarray, band: np.ndarray, char_h: float,
                        side: int, lo: float, keep: np.ndarray | None = None) -> np.ndarray:
    """16 方向匹配线滤波恢复弱墨迹，法向背景相减抑制笔画两侧灰晕。

    在线结构的灰度上积累证据，不要求断笔已经能二值化为骨架。
    每行独立处理，保留纸面噪声门槛；无墨迹的长空白不会生成笔画。
    """
    valid = band.astype(bool).copy()
    if keep is not None:
        valid &= keep == 0
    result = np.zeros(bh.shape, np.float32)
    if not valid.any():
        return result
    scale = float(np.clip(side / 2480, 0.6, 2.0))
    length = float(np.clip(0.075 * char_h, 2.0 * scale, 4.5 * scale))
    normal_offset = 2.0 * scale
    radius = int(np.ceil(length + 2))
    size = 2 * radius + 1
    kernels = []
    for theta in np.linspace(0, np.pi, 16, endpoint=False):
        dx, dy = np.cos(theta), np.sin(theta)
        kernel = np.zeros((size, size), np.float32)
        for t in np.linspace(-length, length, 65):
            x, y = radius + t * dx, radius + t * dy
            ix, iy = int(x), int(y)
            fx, fy = x - ix, y - iy
            weight = np.exp(-0.5 * (t / max(length / 2, 1)) ** 2)
            kernel[iy:iy+2, ix:ix+2] += weight * np.array(
                [[(1-fx)*(1-fy), fx*(1-fy)], [(1-fx)*fy, fx*fy]], np.float32)
        kernel /= kernel.sum()
        yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
        projection = xx * dx + yy * dy
        half_left = kernel * (projection < -0.5)
        half_right = kernel * (projection > 0.5)
        half_left /= half_left.sum()
        half_right /= half_right.sum()
        kernels.append((kernel, half_left, half_right, normal_offset * dy, -normal_offset * dx))
    active = valid.any(axis=1)
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    win = max(3, int(0.45 * char_h) | 1)
    pad = int(np.ceil(0.7 * char_h + radius + normal_offset))
    floor = max(1.2, 0.45 * lo)
    for start, end in zip(starts, ends):
        columns = np.flatnonzero(valid[start:end].any(axis=0))
        x0, x1 = max(0, int(columns[0])-pad), min(bh.shape[1], int(columns[-1])+pad+1)
        y0, y1 = max(0, start-pad), min(bh.shape[0], end+pad)
        mask = valid[y0:y1, x0:x1]
        src = np.where(mask, bh[y0:y1, x0:x1], 0).astype(np.float32)
        best = np.zeros_like(src)
        for kernel, half_left, half_right, nx, ny in kernels:
            along = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REFLECT)
            transform = np.array([[1, 0, nx], [0, 1, ny]], np.float32)
            left = cv2.warpAffine(along, transform, (src.shape[1], src.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            transform[:, 2] *= -1
            right = cv2.warpAffine(along, transform, (src.shape[1], src.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ridge = np.maximum(along - 0.5 * (left + right), 0)
            # 空白处只有两侧都存在墨迹才补，避免把端点拖成长尾。
            backward = cv2.filter2D(src, -1, half_left, borderType=cv2.BORDER_REFLECT)
            forward = cv2.filter2D(src, -1, half_right, borderType=cv2.BORDER_REFLECT)
            bilateral = np.minimum(backward, forward)
            ridge[(src <= lo) & (bilateral < np.maximum(lo, 0.4 * along))] = 0
            np.maximum(best, ridge, out=best)
        peak = cv2.GaussianBlur(cv2.dilate(best, np.ones((win, win), np.uint8)),
                                (0, 0), max(1.0, char_h / 5))
        u = np.clip((best - floor) / np.maximum(0.65 * peak - floor, max(2.5, 0.7 * lo)), 0, 1)
        # 原始墨迹附近才允许定向恢复，孤立颗粒不能靠滤波拖出短线。
        evidence = (src > max(lo, 3.0)).astype(np.uint8)
        support = cv2.filter2D(evidence.astype(np.float32), -1,
                               np.ones((3, 3), np.float32)) >= 2
        reach = max(1, int(np.ceil(length)))
        support = cv2.dilate(support.astype(np.uint8), np.ones((2*reach+1, 2*reach+1), np.uint8)) > 0
        u[~mask | ~support] = 0
        # 只回写当前行，避免重叠上下文使不同分块重复增强。
        target = result[start:end, x0:x1]
        np.maximum(target, u[start-y0:end-y0], out=target)
    return result


def interp_strokes(bh: np.ndarray, band: np.ndarray, char_h: float, side: int, lo: float,
                   keep: np.ndarray | None = None) -> np.ndarray:
    """正文行里的细笔画插值：返回 0~1 的加深强度。

    以原始笔画峰值归一化，少量平滑抑制颗粒，保留连续灰阶边缘。
    普通平滑限于已有墨迹的一像素邻域；再由骨架端点匹配修补短断笔，
    避免扩散描粗或合并相邻笔画。
    """
    # 小于一个像素的平滑只抑制扫描颗粒；不再把扩散出来的灰晕阈值化成黑边。
    sigma = max(0.35, min(0.8, 0.55 * side / 2480))
    valid = band.astype(bool).copy()
    if keep is not None:
        valid &= keep == 0
    src = np.where(valid, bh, 0).astype(np.float32)
    smooth = cv2.GaussianBlur(src, (0, 0), sigma)
    win = max(3, int(0.6 * char_h) | 1)
    peak = cv2.GaussianBlur(cv2.dilate(src, np.ones((win, win), np.uint8)),
                            (0, 0), max(1.0, char_h / 3))
    # 用原始笔画的峰值归一化，避免平滑降低峰值后把两侧浅边放大。
    floor = max(ENHANCE_MIN_DARK, lo)
    full = np.maximum(0.55 * peak, floor + 6.0)
    signal = 0.65 * src + 0.35 * smooth
    u = np.clip((signal - floor) / (full - floor), 0, 1)
    # 仅允许在已有墨迹的一像素邻域内微量补缺，不向空白扩散。
    support = cv2.dilate((src > floor).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    added = src <= floor
    u[added] = np.minimum(u[added], 0.25)
    u[~valid | ~support] = 0
    return repair_stroke_gaps(u.astype(np.float32), src, valid, char_h, floor)


def enhance_text(img: np.ndarray, bg: tuple[int, int, int], keep: np.ndarray | None = None,
                 interp: bool = True, text_method: str = "directional") -> tuple[np.ndarray, np.ndarray]:
    """把正文行里偏淡的笔画加深到接近黑色，返回 (增强后的图, 宽正文行的笔画掩码，供去噪保护)。

    复印件墨粉不足时字迹很淡（只比纸色暗十几个灰度）、笔画还被扫成断续的点。只加深“文字”，不加深噪点：
      - 只加深 find_text 找出的正文行笔画，行外的零星墨点、成片污渍、阴影一概不动；
      - 加深幅度按同一个字附近最深的笔画归一化，淡字、深字最后黑度接近；本来就深的像素不再改动；
      - 只改 Lab 亮度，彩色内容（红头、印章）保持色相；印章保护区不动；
      - interp：启用淡字恢复；text_method 选择定向灰度恢复或旧版骨架端点连接。
    """
    text, wide, band, ext_dark, char_h, bh, ds, lo, n_lines = find_text(img, bg)
    paper = float(np.mean(bg))
    side = min(img.shape[:2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keep is not None:
        text &= keep == 0
        wide &= keep == 0
    if not text.any():
        log.info("  增强: 未发现正文行，跳过")
        return img, wide.astype(np.uint8) * 255

    # 局部笔画对比度：同一个字附近最深的笔画；加深后淡字和深字一样黑
    win = max(3, int(0.6 * char_h) | 1)
    hi = cv2.GaussianBlur(cv2.dilate(ds * text, np.ones((win, win), np.uint8)), (0, 0), char_h / 3)
    hi = np.maximum(hi, 2 * lo)
    # 判断“是不是淡字”时，行延伸出来的地方看源行：清楚的行延伸到的空白、透字不加深
    hi_fade = np.maximum(hi, ext_dark)
    # 连续灰阶映射保留抗锯齿边缘，不做闭运算或低阈值饱和，避免描粗和粘连。
    raw_peak = cv2.GaussianBlur(cv2.dilate(bh * text, np.ones((win, win), np.uint8)),
                                (0, 0), max(1.0, char_h / 3))
    full = np.maximum(0.55 * raw_peak, lo + 6.0)
    t = np.clip((bh - lo) / (full - lo), 0, 1)
    m = text.astype(np.float32)
    # 本来就清楚的字（附近最深的笔画已经够深）逐渐不再加深：只救淡字，不把正常的字描粗
    fade = np.clip((ENHANCE_CLEAR - hi_fade) / (ENHANCE_CLEAR - ENHANCE_FAINT), 0, 1)
    m *= fade
    if keep is not None:
        m[keep > 0] = 0
    v = t * m
    if interp:
        # 插值只给真正很淡的字用：清楚的字插值只会在笔画外描出一圈灰晕
        faint = np.clip((ENHANCE_INTERP_CLEAR - hi_fade) / (ENHANCE_INTERP_CLEAR - ENHANCE_INTERP_FAINT), 0, 1)
        log.info("  淡字恢复: %s", "16方向灰度线滤波" if text_method == "directional" else "骨架端点连接")
        reconstruct = directional_strokes if text_method == "directional" else interp_strokes
        reconstructed = reconstruct(bh, band, char_h, side, lo, keep)
        v = v * (1 - faint) + reconstructed * faint * fade
    g = gray.astype(np.float32)
    # 只往深里改：已经比目标深的像素不动
    drop = np.maximum(g - (paper - v * (paper - ENHANCE_INK)), 0)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_img[..., 0] = np.clip(lab_img[..., 0] - drop, 0, 255)  # OpenCV 8 位 Lab 的 L 也是 0~255
    out = img.copy()
    changed = drop > 0
    converted = cv2.cvtColor(np.round(lab_img).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out[changed] = converted[changed]
    log.info("  增强: 正文行 %d 行，字高 %.0fpx，加深 %.2f%% 的面积（暗度下限 %.1f）",
             n_lines, char_h, 100 * float((drop > 30).mean()), lo)
    return out, wide.astype(np.uint8) * 255


# ---------------------------------------------------------------- 方向检测与旋转校正

def rotate_cw(img: np.ndarray, deg: int) -> np.ndarray:
    """顺时针旋转 0/90/180/270°（无损，90/270° 时宽高互换）。"""
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return img if deg % 360 == 0 else cv2.rotate(img, codes[deg % 360])


def shrink(img: np.ndarray, max_side: int) -> np.ndarray:
    s = max_side / max(img.shape[:2])
    return img if s >= 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run_osd(img: np.ndarray) -> tuple[int, float, str] | None:
    """调用 Tesseract OSD，返回 (需顺时针旋转的角度, 方向置信度, 文字类型)；无法检测时返回 None。"""
    exe = shutil.which("tesseract")
    if exe is None:
        return None
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    try:
        r = subprocess.run([exe, "stdin", "stdout", "--psm", "0"], input=buf.tobytes(),
                           capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("    OSD 调用失败: %s", e)
        return None
    out = r.stdout.decode(errors="replace")
    rot = re.search(r"^Rotate:\s*(\d+)", out, re.M)
    conf = re.search(r"^Orientation confidence:\s*([\d.]+)", out, re.M)
    script = re.search(r"^Script:\s*(\S+)", out, re.M)
    if rot is None or conf is None:
        err = r.stderr.decode(errors="replace").strip().splitlines()
        log.debug("    OSD 无结果: %s", err[-1] if err else "(无输出)")
        return None
    return int(rot.group(1)) % 360, float(conf.group(1)), script.group(1) if script else "?"


def projection_sharpness(profile: np.ndarray) -> float:
    """投影曲线的“锐度”：相邻差值平方和（按均值归一化）。文字行与行间空白交替时最大。"""
    p = profile.astype(np.float64)
    return float(np.sum(np.diff(p) ** 2) / (np.sum(p) ** 2 / len(p) + 1e-9))


def text_line_ratio(img: np.ndarray) -> float:
    """文字行方向比值：行投影锐度 / 列投影锐度。横排文字 >1，竖排（或横躺的横排）<1。

    先用开运算去掉表格长直线，只看文字笔画。
    """
    g = to_gray(shrink(img, 1200))
    ink = (g < float(np.median(g)) * 0.65).astype(np.uint8)
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 41)))
    text = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.bitwise_or(hl, vl)))
    if not text.any():
        return 1.0
    return projection_sharpness(text.sum(axis=1)) / max(projection_sharpness(text.sum(axis=0)), 1e-9)


def detect_orientation(img: np.ndarray) -> tuple[int, bool]:
    """检测把文字转正需要顺时针旋转的角度。返回 (角度, 方向是否已确认)。

    为杜绝误转：
      - 原图检测为 0° 时直接保持原样（不转永远不会“转错”）；
      - 检测到需要旋转时，再把图分别转 90/180/270° 各测一次，四次推出的角度必须完全一致、
        且每次置信度都不低于 OSD_MIN_CONF；
      - 90/270° 旋转后文字行必须不呈竖排（text_line_ratio ≥ ROTATE_VETO）。
    任一条件不满足都保持原方向。
    """
    if shutil.which("tesseract") is None:
        log.warning("  方向: 未找到 tesseract，跳过方向检测（brew install tesseract）")
        return 0, False
    small = shrink(img, OSD_MAX_SIDE)
    first = run_osd(small)
    if first is None:
        log.info("  方向: 文字过少或无法识别，保持原方向")
        return 0, False
    rot, conf, script = first
    log.debug("  方向: OSD 原图 → 需旋转 %d°（置信度 %.2f，文字 %s）", rot, conf, script)
    if rot == 0:
        if conf >= OSD_MIN_CONF:
            log.info("  方向: 已朝上（置信度 %.2f，文字 %s）", conf, script)
            return 0, True
        log.info("  方向: 判为朝上但置信度低（%.2f < %.2f），保持原方向", conf, OSD_MIN_CONF)
        return 0, False

    # 复核：图像先顺时针转 t°，OSD 给出 r，则原图应转 (t + r) % 360
    votes = [(0, rot, conf)]
    for t in (90, 180, 270):
        res = run_osd(rotate_cw(small, t))
        if res is None:
            log.warning("  方向: 原图判为需旋转 %d°，但旋转 %d° 的副本无法检测，保持原方向", rot, t)
            return 0, False
        votes.append((t, (t + res[0]) % 360, res[1]))
    detail = "，".join(f"副本{t}°→{r}°({c:.2f})" for t, r, c in votes)
    if any(r != rot for _, r, _ in votes):
        log.warning("  方向: 四个方向复核结果不一致（%s），为避免误转，保持原方向", detail)
        return 0, False
    if min(c for _, _, c in votes) < OSD_MIN_CONF:
        log.warning("  方向: 复核一致但置信度偏低（%s），为避免误转，保持原方向", detail)
        return 0, False
    if rot in (90, 270):
        ratio = text_line_ratio(rotate_cw(small, rot))
        if ratio < ROTATE_VETO:
            log.warning("  方向: OSD 判为需旋转 %d°，但旋转后文字行呈竖排（行方向比 %.2f < %.2f），保持原方向",
                        rot, ratio, ROTATE_VETO)
            return 0, False
    log.info("  方向: 需顺时针旋转 %d°（四向复核一致：%s，文字 %s）", rot, detail, script)
    return rot, True


def _skew_search(ink: np.ndarray) -> tuple[float, float, float]:
    """在 ±SKEW_RANGE 内搜索使行投影最锐利的角度。返回 (角度, 相对 0° 的增益, 相对远处次峰的峰值比)。"""
    h, w = ink.shape

    def score(a: float) -> float:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST)
        return float(np.sum(np.diff(r.sum(axis=1).astype(np.float64)) ** 2))

    coarse = np.arange(-SKEW_RANGE, SKEW_RANGE + 1e-6, 0.5)
    cs = np.array([score(a) for a in coarse])
    a0 = float(coarse[int(cs.argmax())])
    fine = np.arange(a0 - 0.75, a0 + 0.75 + 1e-6, 0.05)
    fs = np.array([score(a) for a in fine])
    peak = float(fs.max())
    # 复印件各处表格线角度常略有差异，峰顶是一段平台；取平台（≥90% 峰值）的加权中心，比单个最高点稳定
    top = fs >= 0.9 * peak
    best = float(np.average(fine[top], weights=fs[top]))
    far = cs[np.abs(coarse - best) > 1.5]
    return best, peak / max(score(0.0), 1e-9), peak / max(float(far.max()) if far.size else 0.0, 1e-9)


def estimate_skew(img: np.ndarray, paper: float) -> float:
    """估计小角度倾斜（逆时针为正）。结果不可靠或无需校正时返回 0。"""
    results = []
    for side in (1200, 800):  # 两种分辨率各估一次，结果必须一致
        g = to_gray(shrink(img, side))
        ink = (g < paper * 0.65).astype(np.uint8)
        if not ink.any():
            log.info("  倾斜: 未发现墨迹，不校正")
            return 0.0
        results.append(_skew_search(ink))
    (a1, gain, peak), (a2, _, _) = results
    info = f"{a1:+.2f}°（复核 {a2:+.2f}°，增益 {gain:.2f}，峰值比 {peak:.2f}）"
    if abs(a1) < SKEW_MIN:
        log.info("  倾斜: %s，小于 %.2f°，无需校正", info, SKEW_MIN)
        return 0.0
    reasons = []
    if gain < SKEW_MIN_GAIN:
        reasons.append(f"相对不旋转改善不明显(<{SKEW_MIN_GAIN})")
    if peak < SKEW_MIN_PEAK:
        reasons.append(f"峰值不突出(<{SKEW_MIN_PEAK})")
    if abs(a1 - a2) > SKEW_AGREE:
        reasons.append(f"两种分辨率结果相差超过 {SKEW_AGREE}°")
    if abs(a1) > SKEW_RANGE - 0.5:
        reasons.append("落在搜索范围边界")
    if reasons:
        log.info("  倾斜: %s，%s，为避免误转不校正", info, "；".join(reasons))
        return 0.0
    log.info("  倾斜: %s，校正", info)
    return a1


def content_mask(img: np.ndarray, paper: float) -> np.ndarray:
    """比 ink 更敏感的“内容”掩码：浅色笔迹、彩色印章也算，用于判断旋转会不会切掉内容。"""
    gray = to_gray(img)
    chroma = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    m = ((gray < paper - 40) | (chroma > 60)).astype(np.uint8) * 255
    return cv2.medianBlur(m, 3)  # 去掉 JPEG 噪声造成的单个像素


def deskew(img: np.ndarray, angle: float, bg: tuple[int, int, int], keep: np.ndarray,
           cleaned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 angle（逆时针为正）旋转校正，空角用周围纸张纹理修补。
    返回 (校正后的图, 同样变换后的印章保护掩码, 同样变换后的已清除黑边区域)。

    始终保持原尺寸（输出尺寸须与输入一致）：整体平移到转出画布的内容最少；剩下转出去的只能是
    本来就被图片边缘截断的墨迹（截断的字、扫描残迹、骑缝章），否则宁可不校正。
    """
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    content = content_mask(img, float(np.mean(bg))) & cv2.bitwise_not(keep)
    n, lab, st, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
    x, y, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    # 本来就被图片边缘截断的墨迹（截断的字、扫描残迹）和比标点还小的零星墨点再转出去一点不算丢内容
    spare = (x == 0) | (y == 0) | (x + w == W) | (y + h == H) | (st[:, 4] < max(30, int(H * W * 1e-5)))
    ys, xs = np.nonzero(content)
    whole = ~spare[lab[ys, xs]]
    nx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
    ny = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
    shift = [0, 0]
    for k, (v, size_) in enumerate(((nx, W), (ny, H))):
        if not len(v):
            continue
        cand = np.arange(int(np.floor(min(0.0, size_ - 1 - v.max()))), int(np.ceil(max(0.0, -v.min()))) + 1)

        def lost(vv: np.ndarray) -> np.ndarray:  # 每个平移量下转出画布的像素数
            sv = np.sort(vv)
            return np.searchsorted(sv, -cand, "left") + len(sv) - np.searchsorted(sv, size_ - 1 - cand, "right")
        # 先保完整内容，再少丢截断的墨迹，最后平移越小越好
        lw, la = lost(v[whole]), lost(v)
        d = int(cand[np.lexsort((np.abs(cand), la, lw))[0]])
        shift[k] = d
        M[k, 2] += d
        v += d
    gone = (nx < 0) | (nx > W - 1) | (ny < 0) | (ny > H - 1)
    out_whole, out_any = int((gone & whole).sum()), int(gone.sum())
    if out_whole:
        log.info("  倾斜: 校正会把 %d 个完整内容的像素转出画布，为保持尺寸且不丢内容，不校正", out_whole)
        return img, keep, cleaned
    log.info("  倾斜: 平移 (%d, %d) 像素，转出画布的只有 %d 个贴边截断墨迹、零星墨点的像素", *shift, out_any)
    size = (W, H)
    out = cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=bg)
    # 旋转后露出的空角（原图之外的区域），外扩几个像素覆盖插值造成的接缝
    valid = cv2.warpAffine(np.full((H, W), 255, np.uint8), M, size, flags=cv2.INTER_NEAREST, borderValue=0)
    empty = cv2.dilate(cv2.bitwise_not(valid), np.ones((5, 5), np.uint8))
    log.info("  空角: 修补 %.2f%% 的面积", 100 * float(np.count_nonzero(empty)) / empty.size)
    keep = cv2.warpAffine(keep, M, size, flags=cv2.INTER_NEAREST, borderValue=0)
    # 空角也算清除过的边缘区域：贴着它的笔迹同样是被截断的
    cleaned = cv2.warpAffine(cleaned, M, size, flags=cv2.INTER_NEAREST, borderValue=255)
    return inpaint(out, empty, keep), keep, cleaned


# ---------------------------------------------------------------- 主流程

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process(path: Path, args: argparse.Namespace) -> bool:
    img = imread(path)
    if img is None:
        log.error("无法读取图片: %s", path)
        return False
    H, W = img.shape[:2]
    log.info("[%s] %dx%d", path.name, W, H)
    bg = dominant_color(img, np.full(img.shape[:2], 255, np.uint8))
    paper = float(np.mean(bg))
    log.debug("  纸张底色 BGR=%s", bg)

    # 0. 大面积浅色彩色噪点（粉色墨粉斑、彩色竖条纹等）：先清掉，免得被后续步骤当成墨迹
    img = color_noise_clean(img, bg)

    # 1. 黑边与边框带：基于原始扫描坐标（扫描仪的边框与图片边缘平行），必须在旋转之前处理
    # 骑缝章先保护起来，后续所有清除步骤都不得改动；掩码只在原图上检测一次，之后随图像一起旋转
    keep = trim_keep_at_border(stamp_mask(img, bg), img, bg, args.thresh, args.margin)
    dark = cv2.bitwise_and(border_dark_mask(img, bg, args.thresh, args.margin, keep), cv2.bitwise_not(keep))
    border = cv2.bitwise_and(cv2.bitwise_or(dark, edge_band_mask(img, bg)), cv2.bitwise_not(keep))
    log.info("  黑边: 修补 %.2f%% 的面积", 100 * float(np.count_nonzero(border)) / border.size)
    out = inpaint(img, border, keep)
    # 黑边会截断页边的笔迹，第 4 步据此判断贴边笔迹是不是残迹（边框带只是亮度渐变，不截断笔迹）
    cleaned = dark

    # 2/3. 方向与倾斜
    confirmed = False
    if args.rotate:
        rot, confirmed = detect_orientation(out)
        if rot:
            out, keep, cleaned = rotate_cw(out, rot), rotate_cw(keep, rot), rotate_cw(cleaned, rot)
        angle = estimate_skew(out, paper)
        if angle:
            out, keep, cleaned = deskew(out, angle, bg, keep, cleaned)
    else:
        log.info("  方向/倾斜: 已关闭")

    # 4. 淡字加深：在去噪之前做，去噪才能按完整的字确定版心和字高，不会把淡字当成淡色痕迹抹掉
    if args.enhance:
        out, text = enhance_text(out, bg, keep, args.interp, args.text_method)
    else:
        text = np.zeros(out.shape[:2], np.uint8)
        log.info("  增强: 已关闭")

    # 5. 版心外噪点
    if not args.denoise:
        log.info("  去噪: 已关闭")
    else:
        ratio = text_line_ratio(out)
        if not confirmed and ratio < VERTICAL_TEXT:
            # 版心按横排文字行定位；方向未确认又像竖排时，正文可能被当成版心外噪点，宁可不去噪
            log.warning("  去噪: 方向未确认且文字行不呈横排（行方向比 %.2f < %.2f），为避免误删内容，跳过",
                        ratio, VERTICAL_TEXT)
        else:
            # 正文行里的笔画（哪怕加深后仍偏淡）和印章一样不许去噪改动
            out = inpaint(out, margin_noise_mask(out, bg, args.noise, cleaned), cv2.bitwise_or(keep, text))

    OUTPUT_DIR.mkdir(exist_ok=True)
    dst = OUTPUT_DIR / path.name
    imwrite(dst, out)
    oH, oW = out.shape[:2]
    log.info("  已保存: %s（%dx%d%s）", dst, oW, oH, "，尺寸不变" if (oW, oH) == (W, H) else "")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描件清理：去黑边、方向校正、去噪点，结果输出到 output 目录")
    ap.add_argument("path", type=Path, help="图片路径，或包含图片的目录（不递归）")
    ap.add_argument("-t", "--thresh", type=int, default=80, help="黑边灰度阈值(默认80)")
    ap.add_argument("-m", "--margin", type=int, default=4, help="黑边清除后向外多扩的像素(默认4)")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="关闭方向检测与倾斜校正（默认开启，需安装 tesseract）")
    ap.add_argument("--no-denoise", dest="denoise", action="store_false", help="关闭版心外去噪点（默认开启）")
    ap.add_argument("--no-enhance", dest="enhance", action="store_false", help="关闭正文淡字加深（默认开启）")
    ap.add_argument("--no-interp", dest="interp", action="store_false",
                    help="关闭淡字的细笔画插值和断笔修复，只按像素加深（默认开启）")
    ap.add_argument("--text-method", choices=("directional", "endpoint"), default="directional",
                    help="淡字恢复算法：directional 定向灰度恢复（默认），endpoint 旧版骨架端点连接")
    ap.add_argument("-n", "--noise", type=float, default=NOISE_RATIO,
                    help=f"噪点阈值：孤立墨点最大尺寸占字高的比例，越大清除越多(默认{NOISE_RATIO})")
    ap.add_argument("-v", "--verbose", action="store_true", help="输出详细检测日志")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if not args.path.exists():
        log.error("路径不存在: %s", args.path)
        return 1
    if args.noise <= 0:
        log.error("噪点阈值必须大于 0: %s", args.noise)
        return 1
    if args.path.is_dir():
        files = sorted(p for p in args.path.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
        if not files:
            log.error("目录中没有图片: %s", args.path)
            return 1
    else:
        files = [args.path]

    failed = [p for p in files if not process(p, args)]
    if len(files) > 1:
        log.info("完成: 成功 %d 张，失败 %d 张", len(files) - len(failed), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
