"""扫描件清理：去黑边、方向检测与旋转校正、去页边噪点。

处理顺序：
  1. 去除贴边黑边、黑角与页边“边框带”（按原始扫描坐标处理）
  2. 方向检测（0/90/180/270°，Tesseract OSD），把文字转到朝上
  3. 小角度倾斜校正，旋转后露出的空角用周围纸张纹理修补
  4. 去除版心之外的零星噪点

“宁可不转，也不错转”：方向需在四个旋转副本上检测一致、且与文字行方向不矛盾才旋转；
倾斜角需峰值明显、两种分辨率下结果一致才校正；校正若会把内容转出画布，则改为扩展画布。
不旋转、不倾斜校正时输出尺寸与原图一致。

用法: python pic.py "图片路径或目录" [-t 阈值] [-m 扩展像素] [--no-rotate] [--no-denoise] [-n 噪点阈值] [-v]
结果保存到 ./output 目录，文件名与原图相同。
"""
import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"
log = logging.getLogger("pic")

# 噪点阈值默认值：孤立墨点的最大尺寸占典型字高的比例
NOISE_RATIO = 0.43

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


def border_dark_mask(img: np.ndarray, bg: tuple[int, int, int], thresh: int, margin: int) -> np.ndarray:
    """与图片边缘相连的黑色区域（黑边、三角黑角、撕裂的黑块）的掩码。

    只处理“接触图片边缘”的深色连通域，页面内部的手写、印刷文字不会被触及。
    """
    H, W = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    dark = (gray < thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    min_area = max(30, int(H * W * 0.00005))
    seed = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if touches and (area >= min_area or min(w, h) <= 6):  # 贴边的细碎黑线也要清
            seed[lab == i] = 255
    # 最外圈几个像素几乎一定是扫描边缘，偏暗的残留一并清掉
    f = 4
    frame = np.zeros((H, W), np.uint8)
    frame[:f], frame[-f:], frame[:, :f], frame[:, -f:] = 255, 255, 255, 255
    seed |= frame & (gray < int(np.mean(bg)) - 40).astype(np.uint8) * 255
    if not seed.any():
        return seed
    # 沿渐变的灰色过渡带向外生长（受限于较亮的灰度上限），吃掉黑边的抗锯齿/阴影边缘
    soft = (gray < min(int(np.mean(bg)) - 40, 190)).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(max(margin, 1) * 3):
        seed = cv2.bitwise_and(cv2.dilate(seed, k3), np.maximum(soft, seed))
    core = cv2.dilate(seed, np.ones((margin * 2 + 1,) * 2, np.uint8))
    # 黑边内侧通常还有一圈偏亮的扫描过曝带或残余暗线，按图像大小再向外多扩一些一并修补
    r = max(margin, int(round(0.022 * min(H, W))))
    grown = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    # 但紧挨黑边的手写/印刷字符不能被扩展区吞掉：保护扩展区里“像字”的墨迹块（细长线条不保护）
    ink = ((gray < int(np.mean(bg)) * 0.65) & (core == 0)).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    protect = np.zeros((H, W), np.uint8)
    for i in range(1, n2):
        x, y, w, h, area = st2[i]
        line_like = min(w, h) <= 6 and max(w, h) > 8 * min(w, h)
        if area >= 30 and not line_like and (grown[y:y + h, x:x + w] > 0).any():
            protect[y:y + h, x:x + w][lab2[y:y + h, x:x + w] == i] = 255
    protect = cv2.dilate(protect, np.ones((5, 5), np.uint8))
    return cv2.bitwise_or(core, cv2.bitwise_and(grown, cv2.bitwise_not(protect)))


def margin_noise_mask(img: np.ndarray, bg: tuple[int, int, int], noise_ratio: float = NOISE_RATIO) -> np.ndarray:
    """版心之外的阴影条带与零星噪点的掩码，不含正文。

    noise_ratio: 孤立墨点的最大尺寸占典型字高的比例，越大清除得越多。
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

    # 1. 把字符横向合并成行，用墨迹量足够大的连通域确定版心
    merge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 25, 3), max(H // 200, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, merge), connectivity=8)
    ink_cnt = np.bincount(lab.ravel().astype(np.int32, copy=False), weights=ink.ravel(), minlength=n)[1:]
    if ink_cnt.size == 0 or ink_cnt.max() == 0:
        log.info("  去噪: 未发现墨迹，跳过")
        return np.zeros((H, W), np.uint8)
    # 墨迹量至少相当于一个字的才算正文行；不用“相对最大值”的比例，否则一张大照片会把所有文字行都筛掉
    big = np.flatnonzero(ink_cnt >= 0.3 * char_h * char_h) + 1
    boxes = []  # 用墨迹本身的外框（而非膨胀后的）判断，避免细黑线因膨胀被当成正文
    for i in big:
        ys, xs = np.nonzero((lab == i) & (ink > 0))
        if ys.size and min(np.ptp(xs), np.ptp(ys)) + 1 > H * 0.008:
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        log.info("  去噪: 未找到正文行，无法确定版心，跳过")
        return np.zeros((H, W), np.uint8)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad_x, pad_y = int(W * 0.02), int(H * 0.015)
    tx0, ty0, tx1, ty1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    log.debug("  去噪: 典型字高 %.0fpx，版心 (%d,%d)-(%d,%d)", char_h, max(tx0, 0), max(ty0, 0),
              min(tx1, W), min(ty1, H))

    outside = np.ones((H, W), np.uint8)
    outside[max(ty0, 0):ty1, max(tx0, 0):tx1] = 0
    kill = np.zeros((H, W), np.uint8)

    # 2. 版心之外的孤立小块墨点：尺寸远小于一个字、接近“团块”形状，并且周围没有其他墨迹
    #    （细长的手写笔画如“1”不删；周围有一串字符的是正文，哪怕颜色浅、没被划进版心也不动）
    # 噪点必须明显小于一个字：孤立的整字（如页码“3”、加粗的标题字）不会被误删
    speck = max(6, int(noise_ratio * char_h))
    radius = 0.05 * min(H, W)
    near = int(0.03 * min(H, W))  # 黑边清除后，被截断的笔迹离边缘仍会留有一小段距离
    n_edge = n_speck = 0
    for i in range(1, n2):
        w, h = st2[i, 2], st2[i, 3]
        cx, cy = cen2[i]
        if not outside[int(min(cy, H - 1)), int(min(cx, W - 1))]:
            continue
        x, y = st2[i, 0], st2[i, 1]
        # 版心之外、紧贴（或被截断于）图片边缘的笔迹：多半是相邻页透过来的字或扫描残留，直接去掉
        if x <= near or y <= near or x + w >= W - near or y + h >= H - near:
            kill[lab2 == i] = 255
            n_edge += 1
            log.debug("    清除贴边笔迹 (%d,%d) %dx%d", x, y, w, h)
            continue
        # 接近字大小的块只有“实心墨团”才算噪点（笔画构成的字填充率低）
        solid = st2[i, 4] >= 0.45 * w * h
        if max(w, h) > (2 * speck if solid else speck) or max(w, h) > 3 * min(w, h):
            continue
        dist = np.hypot(cen2[1:, 0] - cx, cen2[1:, 1] - cy)
        if int((dist < radius).sum()) - 1 > 2:  # 减去自己；邻居多说明是文字
            continue
        kill[lab2 == i] = 255
        n_speck += 1
        log.debug("    清除孤立墨点 (%d,%d) %dx%d", x, y, w, h)

    log.info("  去噪: 清除贴边笔迹 %d 处、孤立墨点 %d 处（墨点尺寸上限 %dpx = 字高 %.0fpx × %.2f）",
             n_edge, n_speck, speck, char_h, noise_ratio)
    return cv2.dilate(kill, np.ones((7, 7), np.uint8))


def edge_band_mask(img: np.ndarray) -> np.ndarray:
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
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用周围纸张纹理修补掩码区域，并在边界羽化过渡，避免出现色块和明显分界。"""
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
    small = cv2.resize(img, (int(W / scale), int(H / scale)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(work, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    fixed = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    fixed = cv2.resize(fixed, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # 修补结果本身较平滑，叠加原图的高频纸纹噪声的缺失在此可忽略；按羽化后的掩码混合
    sigma = max(1.0, 0.002 * min(H, W))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)[..., None]
    alpha = np.maximum(alpha, (mask > 0)[..., None].astype(np.float32))
    return np.clip(img * (1 - alpha) + fixed * alpha, 0, 255).astype(np.uint8)


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


def deskew(img: np.ndarray, angle: float, bg: tuple[int, int, int]) -> np.ndarray:
    """按 angle（逆时针为正）旋转校正，空角用周围纸张纹理修补。

    默认保持原尺寸；若这样会把任何内容像素转出画布，则扩展画布以完整保留内容。
    """
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    ys, xs = np.nonzero(content_mask(img, float(np.mean(bg))))
    nx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
    ny = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
    lost = int(((nx < 0) | (nx > W - 1) | (ny < 0) | (ny > H - 1)).sum())
    size = (W, H)
    if lost:
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nW, nH = int(np.ceil(W * c + H * s)), int(np.ceil(W * s + H * c))
        M[0, 2] += (nW - W) / 2
        M[1, 2] += (nH - H) / 2
        size = (nW, nH)
        log.info("  倾斜: 原尺寸旋转会切掉 %d 个内容像素，改为扩展画布 %dx%d → %dx%d", lost, W, H, nW, nH)
    out = cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=bg)
    # 旋转后露出的空角（原图之外的区域），外扩几个像素覆盖插值造成的接缝
    valid = cv2.warpAffine(np.full((H, W), 255, np.uint8), M, size, flags=cv2.INTER_NEAREST, borderValue=0)
    empty = cv2.dilate(cv2.bitwise_not(valid), np.ones((5, 5), np.uint8))
    log.info("  空角: 修补 %.2f%% 的面积", 100 * float(np.count_nonzero(empty)) / empty.size)
    return inpaint(out, empty)


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

    # 1. 黑边与边框带：基于原始扫描坐标（扫描仪的边框与图片边缘平行），必须在旋转之前处理
    border = cv2.bitwise_or(border_dark_mask(img, bg, args.thresh, args.margin), edge_band_mask(img))
    log.info("  黑边: 修补 %.2f%% 的面积", 100 * float(np.count_nonzero(border)) / border.size)
    out = inpaint(img, border)

    # 2/3. 方向与倾斜
    confirmed = False
    if args.rotate:
        rot, confirmed = detect_orientation(out)
        if rot:
            out = rotate_cw(out, rot)
        angle = estimate_skew(out, paper)
        if angle:
            out = deskew(out, angle, bg)
    else:
        log.info("  方向/倾斜: 已关闭")

    # 4. 版心外噪点
    if not args.denoise:
        log.info("  去噪: 已关闭")
    else:
        ratio = text_line_ratio(out)
        if not confirmed and ratio < VERTICAL_TEXT:
            # 版心按横排文字行定位；方向未确认又像竖排时，正文可能被当成版心外噪点，宁可不去噪
            log.warning("  去噪: 方向未确认且文字行不呈横排（行方向比 %.2f < %.2f），为避免误删内容，跳过",
                        ratio, VERTICAL_TEXT)
        else:
            out = inpaint(out, margin_noise_mask(out, bg, args.noise))

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
