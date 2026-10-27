import argparse
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import pic
from pic import border_dark_mask, deskew


BG = (255, 255, 255)


class BorderTests(unittest.TestCase):
    def page(self):
        return np.full((3000, 2100, 3), 255, np.uint8)  # 约 A4 250dpi

    def test_table_touching_edge_is_kept(self):
        img = self.page()
        for y in range(300, 2700, 130):  # 横线一直伸到右边缘
            cv2.line(img, (100, y), (2099, y), (0, 0, 0), 4)
        for x in (100, 900, 1500):
            cv2.line(img, (x, 300), (x, 2640), (0, 0, 0), 4)
        img[330:370, 2092:] = 0  # 被右边缘截断的序号“1”
        mask = border_dark_mask(img, BG, 80, 4)
        self.assertFalse(mask[300:2700, 100:2040].any())
        self.assertFalse(mask[330:370, 2092:].any())

    def test_black_border_is_removed(self):
        img = self.page()
        img[:, :60] = 0  # 厚黑边
        img[200:2800, 2096:] = 60  # 沿右边缘的细扫描残线
        mask = border_dark_mask(img, BG, 80, 4)
        self.assertTrue(mask[:, :60].all())
        self.assertTrue(mask[200:2800, 2096:].all())

    def test_deskew_keeps_size(self):
        keep = np.zeros((3000, 2100), np.uint8)
        img = self.page()
        img[300:2700, 10:30] = 0
        img[300:2700, 2070:2090] = 0  # 完整内容转出画布、平移也放不下：不校正
        out, _, _ = deskew(img, 2.0, BG, keep, keep)
        self.assertTrue(np.array_equal(out, img))
        img = self.page()
        img[300:2700, 0:30] = 0  # 只有贴边截断的墨迹会转出去：照常校正
        img[300:2700, 2070:2090] = 0
        out, _, _ = deskew(img, 2.0, BG, keep, keep)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))
        img = self.page()
        img[300:2700, 40:2000] = 0  # 有空白可挪：平移后校正，尺寸不变
        out, _, _ = deskew(img, 2.0, BG, keep, keep)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))


SAMPLE = Path(__file__).resolve().parent.parent / "test" / "1027.png"  # 样张不入库，没有就跳过


@unittest.skipUnless(SAMPLE.exists(), "缺少样张 test/1027.png")
class Sample1027Tests(unittest.TestCase):
    """倒置、略倾斜的签到表：表格线伸到图片边缘，左边序号被截断，空白格里有一片墨粉散点。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        old = pic.OUTPUT_DIR
        pic.OUTPUT_DIR = Path(cls.tmp.name)
        try:
            outs = {}
            for denoise in (True, False):
                args = argparse.Namespace(thresh=80, margin=4, rotate=True, denoise=denoise, enhance=True,
                                          interp=True, text_method="directional", noise=pic.NOISE_RATIO)
                pic.process(SAMPLE, args)
                outs[denoise] = cv2.imread(str(pic.OUTPUT_DIR / SAMPLE.name), cv2.IMREAD_GRAYSCALE)
        finally:
            pic.OUTPUT_DIR = old
        cls.src = cv2.imread(str(SAMPLE), cv2.IMREAD_GRAYSCALE)
        cls.out, cls.raw = outs[True], outs[False]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_size_kept(self):
        self.assertEqual(self.out.shape, self.src.shape)

    def test_table_and_numbers_kept(self):
        self.assertGreater((self.raw < 128).sum(), 0.02 * self.raw.size)  # 表格与签名都在（旧版曾整张清掉）
        # 左侧序号列（含被截断的“1”）不被去噪动
        lost = (self.raw[600:2800, :60] < 128) & (self.out[600:2800, :60] > 200)
        self.assertLess(lost.sum(), 20)

    def test_specks_removed(self):
        # 第 9~15 行的空白格（去掉格线）里的墨粉散点，至少清掉四成
        y0, y1, x0, x1 = 1800, 2750, 420, 2000
        raw, out = self.raw[y0:y1, x0:x1], self.out[y0:y1, x0:x1]
        ink = (raw < 200).astype(np.uint8)
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 300), np.uint8)) \
            | cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((300, 1), np.uint8))
        cells = cv2.dilate(lines, np.ones((15, 15), np.uint8)) == 0
        before, after = ((raw < 128) & cells).sum(), ((out < 128) & cells).sum()
        self.assertGreater(before, 1000)
        self.assertLess(after, 0.6 * before)

if __name__ == "__main__":
    unittest.main()
