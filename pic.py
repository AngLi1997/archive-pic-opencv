"""去除图片黑边（定位内容区域后，黑边用主背景色填充，保持尺寸与方向不变）。

用法: python pic.py "图片路径或目录" [-t 阈值] [-m 扩展像素]
结果保存到 ./output 目录，文件名与原图相同。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"


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
    return tuple(int(v) for v in pixels[flat == top].mean(axis=0))


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
        seed = cv2.dilate(seed, k3) & np.maximum(soft, seed)
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
    return core | (grown & ~protect)


def margin_noise_mask(img: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """版心之外的阴影条带与零星噪点的掩码，不含正文。"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = float(np.mean(bg))
    ink = (gray < paper * 0.65).astype(np.uint8)

    # 1. 把字符横向合并成行，用墨迹量足够大的连通域确定版心
    merge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 25, 3), max(H // 200, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, merge), connectivity=8)
    ink_cnt = np.bincount(lab.ravel(), weights=ink.ravel(), minlength=n)[1:]
    if ink_cnt.size == 0 or ink_cnt.max() == 0:
        return np.zeros((H, W), np.uint8)
    big = np.flatnonzero(ink_cnt >= 0.25 * np.median(ink_cnt[ink_cnt > 0.1 * ink_cnt.max()])) + 1
    boxes = []  # 用墨迹本身的外框（而非膨胀后的）判断，避免细黑线因膨胀被当成正文
    for i in big:
        ys, xs = np.nonzero((lab == i) & (ink > 0))
        if ys.size and min(np.ptp(xs), np.ptp(ys)) + 1 > H * 0.008:
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        return np.zeros((H, W), np.uint8)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad_x, pad_y = int(W * 0.02), int(H * 0.015)
    tx0, ty0, tx1, ty1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y

    outside = np.ones((H, W), np.uint8)
    outside[max(ty0, 0):ty1, max(tx0, 0):tx1] = 0
    kill = np.zeros((H, W), np.uint8)

    # 2. 版心之外的孤立小块墨点：尺寸远小于一个字、接近“团块”形状，并且周围没有其他墨迹
    #    （细长的手写笔画如“1”不删；周围有一串字符的是正文，哪怕颜色浅、没被划进版心也不动）
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(ink, connectivity=8)
    speck = max(8, int(0.03 * min(H, W)))
    radius = 0.05 * min(H, W)
    near = int(0.03 * min(H, W))  # 黑边清除后，被截断的笔迹离边缘仍会留有一小段距离
    for i in range(1, n2):
        w, h = st2[i, 2], st2[i, 3]
        cx, cy = cen2[i]
        if not outside[int(min(cy, H - 1)), int(min(cx, W - 1))]:
            continue
        x, y = st2[i, 0], st2[i, 1]
        # 版心之外、紧贴（或被截断于）图片边缘的笔迹：多半是相邻页透过来的字或扫描残留，直接去掉
        if x <= near or y <= near or x + w >= W - near or y + h >= H - near:
            kill[lab2 == i] = 255
            continue
        if max(w, h) > speck or max(w, h) > 3 * min(w, h):
            continue
        dist = np.hypot(cen2[1:, 0] - cx, cen2[1:, 1] - cy)
        if int((dist < radius).sum()) - 1 > 2:  # 减去自己；邻居多说明是文字
            continue
        kill[lab2 == i] = 255

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
    work = mask | cv2.dilate(ink & near, np.ones((5, 5), np.uint8))
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process(path: Path, thresh: int, margin: int) -> bool:
    img = imread(path)
    if img is None:
        print(f"无法读取图片: {path}", file=sys.stderr)
        return False
    bg = dominant_color(img, np.full(img.shape[:2], 255, np.uint8))
    out = inpaint(img, border_dark_mask(img, bg, thresh, margin) | edge_band_mask(img))
    out = inpaint(out, margin_noise_mask(out, bg))
    OUTPUT_DIR.mkdir(exist_ok=True)
    dst = OUTPUT_DIR / path.name
    imwrite(dst, out)
    print(f"{path.name}: {img.shape[1]}x{img.shape[0]} (尺寸不变)  已保存: {dst}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="去除图片黑边，结果输出到 output 目录")
    ap.add_argument("path", type=Path, help="图片路径，或包含图片的目录（不递归）")
    ap.add_argument("-t", "--thresh", type=int, default=80, help="黑边灰度阈值(默认80)")
    ap.add_argument("-m", "--margin", type=int, default=4, help="黑边清除后向外多扩的像素(默认4)")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"路径不存在: {args.path}", file=sys.stderr)
        return 1
    if args.path.is_dir():
        files = sorted(p for p in args.path.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
        if not files:
            print(f"目录中没有图片: {args.path}", file=sys.stderr)
            return 1
    else:
        files = [args.path]

    failed = [p for p in files if not process(p, args.thresh, args.margin)]
    if len(files) > 1:
        print(f"完成: 成功 {len(files) - len(failed)} 张，失败 {len(failed)} 张")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
