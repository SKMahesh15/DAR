from ultralytics import YOLO
from ultralytics import SAM
import cv2
import numpy as np

class DAR:
    def __init__(self, model_path="/home/skm/Downloads/DAR/models/train9/weights/best.pt", sam_model_path="/home/skm/Downloads/DAR/models/sam2.1_l.pt"):
        self.model = YOLO(model_path)
        self.model_sam = SAM(sam_model_path)

    def bbox(self, image_path):
        pred = self.model.predict(image_path)
        boxes = pred[0].boxes
        img = cv2.imread(image_path)

        # draw rectangle
        best_box = []
        best_conf = 0.5  # threshold
        best_area = 0

        if len(boxes) != 0:
            for box in boxes:
                conf = float(box.conf[0])
                area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                if conf >= best_conf and area > best_area:
                    best_conf = conf
                    best_box = box.xyxy[0].tolist()
                    best_area = area

        # Runs Mobile sam only on the best bounding box with the most confidence and most area. 
        if len(best_box) != 0:
            segment = self.model_sam.predict(image_path, bboxes=best_box, save=True)
        else:
            segment = []
            return best_box, best_conf, best_area, segment

        return best_box, best_conf, best_area, segment

    def map_to_ad(pt, ad, pts):
        h_ad, w_ad = ad.shape[:2]

        # Bounding box in pixel space (same space as pt1, pt2, pt3)
        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])

        u = (pt[0] - x_min) / (x_max - x_min)
        v = (pt[1] - y_min) / (y_max - y_min)
        return [int(u * w_ad), int(v * h_ad)]


if __name__ == "__main__":
    warp = DAR()
    warp.bbox("test_images/image copy 4.png")