import numpy as np


def map_to_ad(all_points, pt, ad_image):
    h_ad, w_ad = ad_image.shape[:2]

    # Bounding box in pixel space (same space as pt1, pt2, pt3)
    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])

    u = (pt[0] - x_min) / (x_max - x_min)
    v = (pt[1] - y_min) / (y_max - y_min)
    return [int(u * w_ad), int(v * h_ad)]

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum (x + y)
    # The bottom-right point will have the largest sum (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference (y - x)
    # The bottom-left will have the largest difference (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect