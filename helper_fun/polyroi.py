#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:34:22 2018

@author: omaier
"""


import cv2
import numpy as np


class polyroi:
    def __init__(self, image, pos=None, max_int=3000):
        self.refPt = []
        if pos is None:
            self.image = cv2.cvtColor(
                np.abs(image[:, :].T / np.max(max_int)).astype(np.float32), cv2.COLOR_GRAY2BGR)
        else:
            self.image = cv2.cvtColor(np.abs(
                image[pos, :, :].T / np.max(max_int)).astype(np.float32), cv2.COLOR_GRAY2BGR)
        self.clone = self.image.copy()

    def handle_mouse(self, event, x, y, flags, param):
          # if the left mouse button was clicked, record the starting
          # (x, y) coordinates and indicate that cropping is being
          # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))

            if len(self.refPt) > 1:
                cv2.line(self.image, self.refPt[-2],
                         self.refPt[-1], (0, 0, 255))
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
    #		refPt.append((x, y))
                # draw a rectangle around the region of interest
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if len(self.refPt) > 1:
                cv2.line(self.image, self.refPt[-1],
                         self.refPt[0], (0, 0, 255))
                return self.refPt

    #      cv2.imshow("ROIimage", image)

    def select_roi(self):
        cv2.namedWindow('ROIimage', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROIimage", self.handle_mouse)
        self.image = self.clone.copy()
        self.refPt = []
        # keep looping until the 'q' key is pressed
        while True:
                # display the image and wait for a keypress
            cv2.imshow("ROIimage", self.image)
            key = cv2.waitKey(33)  # & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = self.clone.copy()
                self.refPt = []
                # if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                return self.refPt
