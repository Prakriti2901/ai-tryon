# aimodel.py
import sys
sys.path.append("C:/Users/Prakriti Aayansh/OneDrive/Desktop/AI-tryOn")

import cv2 as cv
import matplotlib.pyplot as plt
from cmate.cmate_main import CMate
from cmate.segmentation.cloth_extractor import extract_cloth
from posestimator.pose_estimator import PoseEstimator, find_rotation_angle
from cmate.custom_shoulder_locator import get_shoulder_details_mannual

class AIModel:
    def __init__(self, source_img_path, dest_img_path):
        self.source_img_path = source_img_path
        self.dest_img_path = dest_img_path

    def load_and_apply_cmate(self):
        try:
            cmate_model = CMate(self.source_img_path, self.dest_img_path)
            final_img, _ = cmate_model.apply_cloth()
            return final_img
        except Exception as e:
            print(f"Error in applying CMate model: {e}")
            return None

    def visualize_pose(self, img_path):
        try:
            img = cv.imread(img_path)
            pose_estimator = PoseEstimator(img)
            pose_estimator.visualize_pose()
        except Exception as e:
            print(f"Error in visualizing pose: {e}")

    def manual_shoulder_detection(self, cloth_seg_img_path):
        try:
            cloth_seg_img = cv.imread(cloth_seg_img_path)
            shoulder_points, distance = get_shoulder_details_mannual(cloth_seg_img)
            return shoulder_points, distance
        except Exception as e:
            print(f"Error in manual shoulder detection: {e}")
            return None, None
