from ultralytics import YOLO
from ultralytics import SAM
import cv2
import numpy as np
from cropper import Cropper

class DAR:
    def __init__(self, original_image, ad_image):
        self.original_image = original_image
        self.ad_image = ad_image

    def render_ad(self):
        crop = Cropper(self.original_image, self.ad_image)
        crop.warp_ad()


if __name__ == "__main__":
    warp = DAR("bill1.png", "zomato.jpeg")
    warp.render_ad()