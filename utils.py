import os
import sys
import cv2


def get_filename_only(file_path):
    if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
        return os.path.splitext(file_path)[0].split('/')[-1]
    elif sys.platform == 'win32':
        return os.path.splitext(file_path)[0].split('\\')[-1]


def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.5)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated
