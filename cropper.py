import cv2
import numpy as np
import os
from main import DAR

def show_image(img, save=False, imageName=None):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save and imageName != None:
        cv2.imwrite(f"{imageName}.png")

obj = DAR()

# ---- LOAD IMAGES ----
img_path = os.path.join("test_images", "bill1.png")
ad_path = os.path.join("test_images", "McdAd.png")
img = cv2.imread(img_path)
ad  = cv2.imread(ad_path)

h_base, w_base, c_base = img.shape
h_subject, w_subject, c_subject = ad.shape

# ---- DETECTION + SEGMENTATION ----
# best_box, best_conf, best_area, segment = obj.bbox(img_path)
# pts = np.array(segment[0].masks.xy[0])
# mask_binary = segment[0].masks.data[0].cpu().numpy()
# np.save("points.npy", pts)
# np.save("segmentPoints.npy", mask_binary)

pts = np.load("points.npy")
def map_to_ad(pt):
    h_ad, w_ad = ad.shape[:2]

    # Bounding box in pixel space (same space as pt1, pt2, pt3)
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    u = (pt[0] - x_min) / (x_max - x_min)
    v = (pt[1] - y_min) / (y_max - y_min)
    return [int(u * w_ad), int(v * h_ad)]

pts_ad = []
for pt in pts:
    pts_ad.append(map_to_ad(pt))
    
pts_ad = np.array(pts_ad)

def sort_pts(points):
    sorted_pts = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    sorted_pts[0] = points[np.argmin(s)]
    sorted_pts[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    sorted_pts[1] = points[np.argmin(diff)]
    sorted_pts[3] = points[np.argmax(diff)]

    return np.float32(sorted_pts)

pts_n = np.load("points_n.npy")
mask_binary = np.load("segmentPoints.npy")
mask_binary = mask_binary.astype(np.uint8)
if mask_binary.max() == 1:
    mask_binary = mask_binary * 255

mask_img = np.zeros_like(img)
mask_img[mask_binary == 1] = [0,255,0]
show_image(img)
h_base, w_base, c_base = img.shape
h_subject, w_subject = ad.shape[:2]

def map_to_ad(pt):
    h_ad, w_ad = ad.shape[:2]

    # Bounding box in pixel space (same space as pt1, pt2, pt3)
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    u = (pt[0] - x_min) / (x_max - x_min)
    v = (pt[1] - y_min) / (y_max - y_min)
    return [int(u * w_ad), int(v * h_ad)]

def get_mesh(pts, show_img=False):
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]
    rect = (x, y, img.shape[1], img.shape[0])

    subdiv = cv2.Subdiv2D(rect)
    for pt in pts:
        x, y = pt[0], pt[1]
        subdiv.insert((float(x), float(y)))

    triangle_list = subdiv.getTriangleList()
    ad_triangle_list = []

    for t in triangle_list:
        pt1 = [int(t[0]), int(t[1])]
        pt2 = [int(t[2]), int(t[3])]
        pt3 = [int(t[4]), int(t[5])]

        # Bounds check
        if not (0 <= pt1[0] < w_base and 0 <= pt1[1] < h_base and
                0 <= pt2[0] < w_base and 0 <= pt2[1] < h_base and
                0 <= pt3[0] < w_base and 0 <= pt3[1] < h_base):
            continue

        # Centroid inside mask check
        cx = (pt1[0] + pt2[0] + pt3[0]) // 3
        cy = (pt1[1] + pt2[1] + pt3[1]) // 3
        if mask_binary[cy, cx] == 0:
            continue

        # cv2.line(img, pt1, pt2, (0,255,0), 1)
        # cv2.line(img, pt2, pt3, (0,255,0), 1)
        # cv2.line(img, pt3, pt1, (0,255,0), 1) 

        src_pt1 = map_to_ad(pt1)
        src_pt2 = map_to_ad(pt2)
        src_pt3 = map_to_ad(pt3)
        src_tri = np.float32([
            src_pt1,
            src_pt2,
            src_pt3
        ])
        ad_triangle_list.append([src_pt1, src_pt2, src_pt3])
    # show_image(img)
    return triangle_list, np.array(ad_triangle_list)

triangle_pts, ad_triangle_list = get_mesh(pts)

tri_contour = ad_triangle_list.reshape((-1, 1, 2))

print(triangle_pts)
print(ad_triangle_list)

def warp_ad():
    triangle_pts, ad_triangle_list = get_mesh(pts)
    x, y, w, h = cv2.boundingRect(pts)
    h_base, w_base, c_base = img.shape
    h_ad, w_ad = ad.shape[:2]
    roi = img[y:y+h, x:x+w]
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    for t in triangle_pts:
        pt1 = [int(t[0]), int(t[1])]
        pt2 = [int(t[2]), int(t[3])]
        pt3 = [int(t[4]), int(t[5])]

        # Bounds check
        if not (0 <= pt1[0] < w_base and 0 <= pt1[1] < h_base and
                0 <= pt2[0] < w_base and 0 <= pt2[1] < h_base and
                0 <= pt3[0] < w_base and 0 <= pt3[1] < h_base):
            continue

        # Centroid inside mask check
        cx = (pt1[0] + pt2[0] + pt3[0]) // 3
        cy = (pt1[1] + pt2[1] + pt3[1]) // 3
        if mask_binary[cy, cx] == 0:
            continue

        # Billboard triangle (destination)
        dst_tri = np.float32([
            pt1,
            pt2,
            pt3
        ])

        # Ad triangle (source)
        src_pt1 = map_to_ad(pt1)
        src_pt2 = map_to_ad(pt2)
        src_pt3 = map_to_ad(pt3)
        src_tri = np.float32([
            src_pt1,
            src_pt2,
            src_pt3
        ])

        # cv2.line(roi, pt1, pt2, (0,255,0), 1)
        # cv2.line(roi, pt2, pt3, (0,255,0), 1)
        # cv2.line(roi, pt3, pt1, (0,255,0), 1)
        h_roi, w_roi = roi.shape[:2]
        tri_mask_binary = np.zeros((h_roi, w_roi), dtype=np.uint8)
        show_image(tri_mask_binary)
        cv2.fillPoly(tri_mask_binary, [dst_tri.astype(np.int32)], 255)
        show_image(tri_mask_binary)

        dst_tri_local = dst_tri - [x, y]
        M = cv2.getAffineTransform(src_tri, dst_tri_local.astype(np.float32)) # gets perspective matrix
        warped_ad = cv2.warpAffine(ad, M, (w, h))                  # warps according to local triangle
        mask_poly = dst_tri_local.astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_poly], 255)
        cv2.copyTo(warped_ad, mask=mask, dst=roi) 
        img[y:y+h, x:x+w] = roi

    show_image(img)

warp_ad()
