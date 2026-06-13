import cv2
import numpy as np
import os
from main import DAR
from helper import order_points, map_to_ad

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
print(pts)
print(type(pts))

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
h_base, w_base, c_base = img.shape
h_subject, w_subject = ad.shape[:2]


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
        print(pt1)
        print(type(pt1))

        src_pt1 = map_to_ad(pts, pt1, ad)
        src_pt2 = map_to_ad(pts, pt2, ad)
        src_pt3 = map_to_ad(pts, pt3, ad)
        src_tri = np.float32([
            src_pt1,
            src_pt2,
            src_pt3
        ])
        ad_triangle_list.append([src_pt1, src_pt2, src_pt3])
    # show_image(img)
    return triangle_list, np.float32(ad_triangle_list)

def check_inside(w_base, h_base, pt1, pt2, pt3):
    if not (0 <= pt1[0] < w_base and 0 <= pt1[1] < h_base): return False
    if not (0 <= pt2[0] < w_base and 0 <= pt2[1] < h_base): return False
    if not (0 <= pt3[0] < w_base and 0 <= pt3[1] < h_base): return False

    cx = (pt1[0] + pt2[0] + pt3[0]) // 3
    cy = (pt1[1] + pt2[1] + pt3[1]) // 3

    if mask_binary[cy, cx] == 0:
        return False

    return True

triangle_pts, ad_triangle_list = get_mesh(pts)
for t in triangle_pts:
    pt1 = [int(t[0]), int(t[1])]
    pt2 = [int(t[2]), int(t[3])]
    pt3 = [int(t[4]), int(t[5])]
    h_base, w_base, c_base = img.shape
    if check_inside(w_base, h_base, pt1, pt2, pt3):
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        cv2.line(img, pt2, pt3, (255, 0, 0), 2)
        cv2.line(img, pt3, pt1, (255, 0, 0), 2)
show_image(img)

def warp_ad():
    triangle_pts, ad_triangle_list = get_mesh(pts)
    h_ad, w_ad = ad.shape[:2]
    ad_preview = ad.copy()
    show_image(ad)
    x, y, w, h = cv2.boundingRect(pts)
    h_base, w_base, c_base = img.shape
    
    roi = img[y:y+h, x:x+w]
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    for t in triangle_pts:
        pt1 = [int(t[0]), int(t[1])]
        pt2 = [int(t[2]), int(t[3])]
        pt3 = [int(t[4]), int(t[5])]

        # Bounds check
        if check_inside(w_base, h_base, pt1, pt2, pt3):
            # Billboard triangle (destination)
            dst_tri = np.float32([
                pt1,
                pt2,
                pt3
            ])

            # Ad triangle (source)
            src_pt1 = map_to_ad(pts, pt1, ad)
            src_pt2 = map_to_ad(pts, pt2, ad)
            src_pt3 = map_to_ad(pts, pt3, ad)
            src_tri = np.float32([
                src_pt1,
                src_pt2,
                src_pt3
            ])

            
            tri_contour = ad_triangle_list.astype(np.int32).reshape((-1, 2))
            hull_points = cv2.convexHull(tri_contour) # gets the contour of the mesh 

            # Calculates the approx bounding box
            hull_perimeter = cv2.arcLength(hull_points, True) 
            epsilon = 0.03 * hull_perimeter
            approx_vertices = cv2.approxPolyDP(hull_points, epsilon, True)

            pts_src = np.float32([
                [0, 0],
                [w_ad - 1, 0],
                [w_ad - 1, h_ad - 1],
                [0, h_ad - 1]
            ])

            pts_dst = np.array(order_points(approx_vertices.reshape(4,2)))
            center = np.mean(pts_dst, axis=0)

            # 2. Shift to origin, scale up by 5% (1.05), and shift back
            pts_dst_expanded = (pts_dst - center) * 1.05 + center

            # 3. Cast back to int32 or float32 depending on what OpenCV needs next
            pts_dst = pts_dst_expanded.astype(np.float32)

            # warps the ad to the bounding box. 
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            temp_ad = cv2.warpPerspective(ad_preview, matrix, (w_ad, h_ad))

            dst_tri_local = dst_tri - [x, y]
            M = cv2.getAffineTransform(src_tri, dst_tri_local.astype(np.float32)) # gets perspective matrix
            warped_ad = cv2.warpAffine(temp_ad, M, (w, h))  # warps according to local triangle
            mask_poly = dst_tri_local.astype(np.int32) # bringing the points to local ad shape

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [mask_poly], 255)
            cv2.copyTo(warped_ad, mask=mask, dst=roi) 
            img[y:y+h, x:x+w] = roi
    show_image(temp_ad)
    show_image(warped_ad)
    show_image(img)

warp_ad()
