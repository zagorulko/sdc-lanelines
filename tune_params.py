#!/usr/bin/python3
# Based on https://github.com/maunesh/opencv-gui-parameter-tuner/
import argparse
import math
import pprint
from collections import defaultdict

import cv2
import numpy as np

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def fit_lines(lines, height):
    x = np.stack((lines[:, 0], lines[:, 2]), axis=-1).reshape(-1)
    y = np.stack((lines[:, 1], lines[:, 3]), axis=-1).reshape(-1)
    f = np.poly1d(np.polyfit(y, x, 1))
    y1, y2 = min(y), height
    x1, x2 = int(f(y1)), int(f(y2))
    return np.array((x1, y1, x2, y2))

def draw_lane_lines(img, lines, slope_low=0.5, slope_high=1.5, **kwargs):
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope_low <= slope <= slope_high:
                left_lines += [line]
            elif -slope_high <= slope <= -slope_low:
                right_lines += [line]
    left_lines = np.array(left_lines, np.int32).reshape(-1, 4)
    right_lines = np.array(right_lines, np.int32).reshape(-1, 4)

    lanes = []
    if left_lines.size:
        lanes += [fit_lines(left_lines, img.shape[0])]
    if right_lines.size:
        lanes += [fit_lines(right_lines, img.shape[0])]

    draw_lines(img, np.array(lanes), **kwargs)

class LaneLineParamTuner:
    def __init__(self, img, region=True):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.region = region

        cv2.namedWindow('tuner')

        self._level = 0
        def on_level_change(pos):
            self._level = pos
            self._render()
        cv2.createTrackbar('level', 'tuner', self._level, 3, on_level_change)

        self.params = {
            'blur': {'ksize': 5},
            'canny': {'th1': 50, 'th2': 150},
            'hough': {'rho': 1, 'theta_deg': 1, 'threshold': 20,
                       'min_line_len': 20, 'max_line_gap': 1},
            'lanes': {'slope_low': 0.5, 'slope_high': 1.5}
        }
        self._create_trackbar('blur', 'ksize', 20,
                              decode=lambda x: x+((x+1)%2))
        self._create_trackbar('canny', 'th1', 255)
        self._create_trackbar('canny', 'th2', 255)
        self._create_trackbar('hough', 'rho', 20)
        self._create_trackbar('hough', 'theta_deg', 180)
        self._create_trackbar('hough', 'threshold', 500)
        self._create_trackbar('hough', 'min_line_len', 500)
        self._create_trackbar('hough', 'max_line_gap', 200)
        self._create_trackbar('lanes', 'slope_low', 200,
                              decode=lambda x: x * 0.01,
                              encode=lambda x: int(x / 0.01))
        self._create_trackbar('lanes', 'slope_high', 200,
                              decode=lambda x: x * 0.01,
                              encode=lambda x: int(x / 0.01))

        self._render()

        print('Adjust parameters as desired. Press X to continue.')
        while cv2.waitKey(0) not in [ord('x'), ord('X')]:
            pass
        cv2.destroyWindow('tuner')

    def _create_trackbar(self, key, subkey, size, decode=lambda x: x,
                         encode=lambda x: x):
        def on_change(pos):
            self.params[key][subkey] = decode(pos)
            self._render()
        name = '{}/{}'.format(key, subkey)
        cv2.createTrackbar(name, 'tuner', encode(self.params[key][subkey]),
                           size, on_change)

    def _render(self):
        img = cv2.cvtColor(self._get_level_output(), cv2.COLOR_RGB2BGR)
        cv2.imshow('output', img)

    def _get_level_output(self):
        x = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        x = cv2.equalizeHist(x)

        d = self.params['blur']
        x = cv2.GaussianBlur(x, (d['ksize'], d['ksize']), sigmaX=0, sigmaY=0)
        if self._level == 0:
            return x

        d = self.params['canny']
        x = cv2.Canny(x, d['th1'], d['th2'])
        if self._level == 1:
            return x

        if self.region:
            mask = np.zeros_like(x)
            h, w = x.shape[0], x.shape[1]
            vertices = [
                (w/15, h),
                (w/2-w*0.01, h*0.56),
                (w/2+w*0.01, h*0.56),
                (w-w/15, h)
            ]
            vertices = np.array([vertices], dtype=np.int32)
            cv2.fillPoly(mask, vertices, 255)
            x = cv2.bitwise_and(x, mask)

        d = self.params['hough']
        lines = cv2.HoughLinesP(x, d['rho'], d['theta_deg']*(math.pi/180),
                                d['threshold'], np.array([]), d['min_line_len'],
                                d['max_line_gap'])
        if self._level == 2:
            x = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
            draw_lines(x, lines)
            return cv2.addWeighted(self.img, 0.8, x, 1, 0)

        d = self.params['lanes']
        if self._level == 3:
            x = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
            draw_lane_lines(x, lines, slope_low=d['slope_low'],
                            slope_high=d['slope_high'], thickness=10)
            return cv2.addWeighted(self.img, 0.8, x, 1, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-R', '--no-region', action='store_true')
    args = parser.parse_args()

    img = cv2.imread(args.file)
    tuner = LaneLineParamTuner(img, region=not args.no_region)
    pprint.pprint(tuner.params, indent=4)

if __name__ == '__main__':
    main()
