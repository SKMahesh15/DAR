import cv2
import numpy as np
import os
from detect import Detect
from helper import order_points, map_to_ad, check_inside

# ---- LOAD IMAGES ----
class Cropper:
    def __init__(self, image_original, ad_image, detector=None):
        self.image_original = image_original
        self.ad_image = ad_image
        self.detector = detector  # Reuse pre-loaded Detect object
        
        # Initialize the shared variables so all methods can see them
        self.img = None
        self.ad = None
        self.pts = None
        self.mask_binary = None
        self.h_base = 0
        self.w_base = 0
        self.h_ad = 0
        self.w_ad = 0
        self.triangle_pts = None
        self.ad_triangle_list = None
        self.if_found = None

    def process_ad(self):
        # Use paths directly as provided (already fully resolved by caller)
        self.img = cv2.imread(self.image_original)
        self.ad  = cv2.imread(self.ad_image)

        # ---- DETECTION + SEGMENTATION ----
        # Reuse the pre-loaded detector if available, otherwise create a new one
        obj = self.detector if self.detector else Detect(self.image_original)
        best_box, best_conf, best_area, segment = obj.bbox(self.image_original)
        if segment:
            self.if_found = True
            self.pts = np.array(segment[0].masks.xy[0])
            self.mask_binary = segment[0].masks.data[0].cpu().numpy()
            self.mask_binary = self.mask_binary.astype(np.uint8)

            if self.mask_binary.max() == 1:
                self.mask_binary = self.mask_binary * 255

            mask_img = np.zeros_like(self.img)
            mask_img[self.mask_binary == 1] = [0,255,0]
            self.h_base, self.w_base, c_base = self.img.shape
            self.h_ad, self.w_ad = self.ad.shape[:2]

            return self.w_ad, self.h_ad, self.h_base, self.w_base, self.mask_binary, self.if_found
        else:
            print("No Bill board found")
            self.if_found = False
            return self.w_ad, self.h_ad, self.h_base, self.w_base, self.mask_binary, self.if_found

        

    def get_mesh(self, show_img=False):
        x, y, w, h = cv2.boundingRect(self.pts)
        rect = (x, y, self.img.shape[1], self.img.shape[0])

        subdiv = cv2.Subdiv2D(rect)
        for pt in self.pts:
            x_pt, y_pt = pt[0], pt[1]
            subdiv.insert((float(x_pt), float(y_pt)))

        triangle_list = subdiv.getTriangleList()
        ad_triangle_list = []

        for t in triangle_list:
            pt1 = [int(t[0]), int(t[1])]
            pt2 = [int(t[2]), int(t[3])]
            pt3 = [int(t[4]), int(t[5])]

            # Bounds check
            if check_inside(self.w_base, self.h_base, pt1, pt2, pt3, self.mask_binary):
                cv2.line(self.img, pt1, pt2, (0,255,0), 1)
                cv2.line(self.img, pt2, pt3, (0,255,0), 1)
                cv2.line(self.img, pt3, pt1, (0,255,0), 1) 

                src_pt1 = map_to_ad(self.pts, pt1, self.ad)
                src_pt2 = map_to_ad(self.pts, pt2, self.ad)
                src_pt3 = map_to_ad(self.pts, pt3, self.ad)
                ad_triangle_list.append([src_pt1, src_pt2, src_pt3])
            
        self.triangle_pts = triangle_list
        self.ad_triangle_list = np.float32(ad_triangle_list)
        return self.triangle_pts, self.ad_triangle_list

    def warp_ad_to_billboard(self):
        ad_preview = self.ad.copy()
        tri_contour = self.ad_triangle_list.astype(np.int32).reshape((-1, 2))
        hull_points = cv2.convexHull(tri_contour) # gets the contour of the mesh 

        # Calculates the approx bounding box
        hull_perimeter = cv2.arcLength(hull_points, True) 
        epsilon = 0.03 * hull_perimeter
        approx_vertices = cv2.approxPolyDP(hull_points, epsilon, True)
        approx_vertices = np.squeeze(approx_vertices, axis=1)

        pts_src = np.float32([
            [0, 0],
            [self.w_ad - 1, 0],
            [self.w_ad - 1, self.h_ad - 1],
            [0, self.h_ad - 1]
        ])

        pts_dst = np.array(order_points(approx_vertices))
        center = np.mean(pts_dst, axis=0)

        # 2. Shift to origin, scale up by 5% (1.05), and shift back
        pts_dst_expanded = (pts_dst - center) * 1.05 + center
        pts_dst = pts_dst_expanded.astype(np.float32)

        # warps the ad to the bounding box. 
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        temp_ad = cv2.warpPerspective(ad_preview, matrix, (self.w_ad, self.h_ad))

        return temp_ad

    def warp_ad(self):
        # We call get_mesh here safely inside the method
        self.get_mesh()
        x, y, w, h = cv2.boundingRect(self.pts)
        
        roi = self.img[y:y+h, x:x+w]

        temp_ad = self.warp_ad_to_billboard()

        for t in self.triangle_pts:
            pt1 = [int(t[0]), int(t[1])]
            pt2 = [int(t[2]), int(t[3])]
            pt3 = [int(t[4]), int(t[5])]

            # Bounds check
            if check_inside(self.w_base, self.h_base, pt1, pt2, pt3, self.mask_binary):
                # Billboard triangle (destination)
                dst_tri = np.float32([
                    pt1,
                    pt2,
                    pt3
                ])

                # Ad triangle (source)
                src_pt1 = map_to_ad(self.pts, pt1, self.ad)
                src_pt2 = map_to_ad(self.pts, pt2, self.ad)
                src_pt3 = map_to_ad(self.pts, pt3, self.ad)
                src_tri = np.float32([
                    src_pt1,
                    src_pt2,
                    src_pt3
                ])

                dst_tri_local = dst_tri - [x, y]
                M = cv2.getAffineTransform(src_tri, dst_tri_local.astype(np.float32)) # gets perspective matrix
                warped_ad = cv2.warpAffine(temp_ad, M, (w, h))  # warps according to local triangle
                mask_poly = dst_tri_local.astype(np.int32) # bringing the points to local ad shape

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [mask_poly], 255)
                cv2.copyTo(warped_ad, mask=mask, dst=roi) 
                self.img[y:y+h, x:x+w] = roi

        return self.img, warped_ad

if __name__ == "__main__":
    img_path = os.path.join("bill1.png")
    ad_path = os.path.join("zomato.jpeg")
    crop = Cropper(img_path, ad_image=ad_path)
    crop.warp_ad()