import unittest

import cv2
import numpy as np

from pic import directional_strokes, interp_strokes, enhance_text


class DirectionalStrokeTests(unittest.TestCase):
    def test_faint_printed_gaps_improve_over_endpoints(self):
        image = np.zeros((180,1100), np.uint8)
        cv2.putText(image, '0123456789 ABCDEFG', (12,105),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, 30, 2, cv2.LINE_AA)
        truth = image.astype(np.float32)
        broken = truth.copy()
        for x in range(30,1050,31):
            broken[:,x:x+2] = 0
        band = np.ones(image.shape, bool)
        old = interp_strokes(broken, band, 60, 2480, 3)
        new = directional_strokes(broken, band, 60, 2480, 3)
        missing = (truth > 15) & (broken == 0)
        old_recovered = np.count_nonzero((old > .4) & missing)
        new_recovered = np.count_nonzero((new > .4) & missing)
        self.assertGreater(new_recovered, 4 * old_recovered)
        self.assertLess(np.count_nonzero((new > .4) & (truth < 3)), 20)

    def test_blank_uniform_noise_and_isolated_pixel(self):
        band = np.ones((100,160), bool)
        for value in (0, 2):
            src = np.full(band.shape, value, np.float32)
            self.assertFalse(directional_strokes(src, band, 40, 2480, 3).any())
        src = np.zeros(band.shape, np.float32)
        src[50,80] = 30
        self.assertFalse(directional_strokes(src, band, 40, 2480, 3).any())

    def test_mask_protection_and_parallel_lines(self):
        src = np.zeros((100,160), np.float32)
        src[20:80,70] = 30
        src[20:80,76] = 30
        band = np.ones(src.shape, bool)
        keep = np.zeros(src.shape, np.uint8)
        keep[40:50] = 255
        out = directional_strokes(src, band, 40, 2480, 3, keep)
        self.assertFalse(out[keep > 0].any())
        self.assertLess(float(out[30:70,73].max()), .2)
        band[:30] = False
        out = directional_strokes(src, band, 40, 2480, 3)
        self.assertFalse(out[~band].any())

    def test_clear_document_does_not_get_bolder(self):
        image = np.full((1000,1400,3), 255, np.uint8)
        for y in (200,300,400,500,600):
            cv2.putText(image, 'CLEAR 0123456789 DOCUMENT', (100,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (20,20,20), 2, cv2.LINE_AA)
        result, _ = enhance_text(image, (255,255,255))
        before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        self.assertEqual(np.count_nonzero(before < 128), np.count_nonzero(after < 128))
        self.assertFalse(((before == 255) & (after < 128)).any())

    def test_no_tail_beyond_intact_line(self):
        src = np.zeros((100,160), np.float32)
        src[50,40:120] = 30
        out = directional_strokes(src, np.ones(src.shape,bool), 40, 2480, 3)
        self.assertLess(float(out[:, :39].max()), .2)
        self.assertLess(float(out[:, 121:].max()), .2)
        self.assertLess(float(out[47:49].max()), .2)


if __name__ == '__main__':
    unittest.main()
