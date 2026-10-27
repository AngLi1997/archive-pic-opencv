import unittest

import cv2
import numpy as np

from pic import repair_stroke_gaps, interp_strokes, enhance_text


class StrokeRepairTests(unittest.TestCase):
    def repair(self, ink, valid=None):
        if valid is None:
            valid = np.ones(ink.shape, bool)
        return repair_stroke_gaps(ink, ink * 30, valid, 40, 3)

    def test_short_gaps_all_directions(self):
        for delta in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            with self.subTest(delta=delta):
                ink = np.zeros((80, 80), np.float32)
                dx, dy = delta
                for t in list(range(-15, -1)) + list(range(2, 16)):
                    ink[40+t*dy, 40+t*dx] = .8
                out = self.repair(ink)
                self.assertGreater(out[40, 40], .5)
                self.assertTrue(np.all(out >= ink))
                self.assertEqual(cv2.connectedComponents((out > .4).astype(np.uint8))[0], 2)

    def test_parallel_strokes_do_not_merge(self):
        ink = np.zeros((80, 80), np.float32)
        ink[20:60, 38] = .8
        ink[20:60, 42] = .8
        np.testing.assert_array_equal(self.repair(ink), ink)

    def test_long_blank_gap_is_not_invented(self):
        ink = np.zeros((80, 80), np.float32)
        ink[40, 20:38] = .8
        ink[40, 43:60] = .8
        np.testing.assert_array_equal(self.repair(ink), ink)

    def test_protected_gap(self):
        ink = np.zeros((80, 80), np.float32)
        ink[40, 20:39] = .8
        ink[40, 42:60] = .8
        valid = np.ones(ink.shape, bool)
        valid[:, 40] = False
        np.testing.assert_array_equal(self.repair(ink, valid), ink)

    def test_intact_line_and_isolated_dots(self):
        ink = np.zeros((80, 80), np.float32)
        ink[40, 20:60] = .8
        ink[20, 30] = ink[20, 33] = .8
        np.testing.assert_array_equal(self.repair(ink), ink)

    def test_crossing_existing_stroke_is_rejected(self):
        ink = np.zeros((80, 80), np.float32)
        ink[40, 20:38] = .8
        ink[40, 43:60] = .8
        ink[30:50, 40] = .8
        np.testing.assert_array_equal(self.repair(ink), ink)

    def test_thick_stroke_keeps_width(self):
        ink = np.zeros((100, 100), np.float32)
        ink[48:51, 20:48] = .8
        ink[48:51, 51:80] = .8
        out = self.repair(ink)
        self.assertGreater(out[49,49], .4)
        self.assertFalse((out[:47] > .4).any())
        self.assertFalse((out[52:] > .4).any())

    def test_long_gap_requires_faint_evidence(self):
        ink = np.zeros((80, 80), np.float32)
        ink[40, 20:38] = .8
        ink[40, 43:60] = .8
        signal = ink * 30
        signal[40, 38:43] = 2
        out = repair_stroke_gaps(ink, signal, np.ones(ink.shape, bool), 40, 3)
        self.assertGreater(out[40,40], .4)

    def test_known_truth_missing_pixels(self):
        truth = np.zeros((240, 320), np.float32)
        for a, b in [((20,40),(140,40)), ((180,15),(180,110)),
                     ((20,140),(100,220)), ((140,220),(220,140))]:
            cv2.line(truth, a, b, .8, 1)
        broken = truth.copy()
        for x, y in [(60,40), (180,60), (60,180), (180,180)]:
            broken[y-1:y+2,x-1:x+2] = 0
        out = repair_stroke_gaps(broken, broken*30, np.ones(truth.shape, bool), 50, 3)
        missing = (truth > .4) & (broken == 0)
        self.assertEqual(np.count_nonzero(missing), 12)
        self.assertTrue((out[missing] > .4).all())
        self.assertFalse(((truth == 0) & (out > .4)).any())

    def test_clean_printed_text_no_new_dark_pixels(self):
        for scale in (0.6, 1.0, 2.5):
            with self.subTest(scale=scale):
                image = np.zeros((180,1200), np.uint8)
                cv2.putText(image, '0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                            (12,105), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            220, 1, cv2.LINE_AA)
                ink = image.astype(np.float32)/255
                out = repair_stroke_gaps(ink, ink*30, np.ones(ink.shape,bool),
                                         22*scale, 3)
                self.assertFalse(((out > .4) & (ink < .1)).any())

    def test_blank_and_band_and_keep(self):
        bh = np.zeros((80, 80), np.float32)
        band = np.ones(bh.shape, bool)
        self.assertFalse(interp_strokes(bh, band, 40, 2480, 3).any())
        bh[40, 20:60] = 30
        keep = np.zeros(bh.shape, np.uint8)
        keep[:, 40:45] = 255
        out = interp_strokes(bh, band, 40, 2480, 3, keep)
        self.assertFalse(out[keep > 0].any())
        band[:, :30] = False
        out = interp_strokes(bh, band, 40, 2480, 3)
        self.assertFalse(out[~band].any())
        image = np.full((200, 200, 3), 245, np.uint8)
        result, _ = enhance_text(image, (245,245,245))
        np.testing.assert_array_equal(image, result)


if __name__ == '__main__':
    unittest.main()
