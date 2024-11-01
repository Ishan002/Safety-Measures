from ultralytics import YOLO
import cv2 as cv 
from scipy.spatial import distance as dist
import numpy as np 
import torch
import datetime
import time
import torch
import cv2
import _thread
from models.yolo import Model
from utils.google_utils import attempt_download
from utils.torch_utils import select_device
from utils.datasets import letterbox
import argparse
from pathlib import Path
import configparser, os
import threading
import time


def load_model(ckpt_path, conf=0.10, iou=0.25):
    model = torch.hub.load('', 'custom',
                           path_or_model=ckpt_path, source='local', force_reload=True)
    model.conf = conf  # NMS confidence threshold
    model.iou = iou  # NMS IoU threshold
    return model

def crop_person_images(boxes, frame, save_dir):
    save_dir = Path(save_dir)  # Convert save_dir to a Path object if it's a string
    cropped_persons = []
    for box in boxes.data:
        label = int(box[5])
        if label == 4:  # Check if detected object is a person
            x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates
            cropped_img = frame[y1:y2, x1:x2]  # Crop the person from the frame
            cropped_persons.append(cropped_img)

    # Generate unique filenames based on current timestamp
    filenames = [f"person_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg" for i in range(len(cropped_persons))]

    # Save the cropped images to the specified directory
    for cropped_img, filename in zip(cropped_persons, filenames):
        cv.imwrite(str(save_dir / filename), cropped_img)
        
def crop_helmet_images(boxes, frame, save_dir):
    save_dir = Path(save_dir)  # Convert save_dir to a Path object if it's a string
    cropped_persons = []
    for i, box in enumerate(boxes.data):
        label = int(box[5])
        if label == 2:  # Check if detected object is a person
            x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates
            cropped_img = frame[y1:y2, x1:x2]  # Crop the person from the frame
            cropped_persons.append(cropped_img)
            filename = f"person_with_helmet_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
            cv.imwrite(str(save_dir / filename), cropped_img)

        
def draw_boxes(results, img, classes):
    if results.numel() > 0:  # Check if results tensor is not empty
        for pred in results:
            if pred.numel() > 0:  # Check if pred tensor is not empty
                x1, y1, x2, y2 = map(int, pred[:4])  # Get bounding box coordinates
                label_class = int(pred[5])
                label = classes[label_class]
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), thickness=2)
                # Draw class label
                cv2.putText(img, f"{label}: {pred[4]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 1)
    return img
        
def check_violation(image_folder, save_dir_all, save_dir_helmet):
    # Define classes for labeling objects
    classes = ["helmet", "nohelmet", "jumpsuit", "nojumpsuit", "person"]
    
    # Iterate through all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read the image
            img_path = os.path.join(image_folder, filename)
            frame = cv.imread(img_path)
            if frame is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Detect objects
            results = model(frame)
            r = results.pred[0]
            
            # Draw bounding boxes on the original image
            img_with_boxes = draw_boxes(r, frame, classes)
            
            # Save entire detected image with bounding boxes
            detected_img_path = os.path.join(save_dir_all, f"detected_{filename}")
            cv.imwrite(detected_img_path, img_with_boxes)
            
            # Crop person images containing helmets
            crop_person_images(r, frame, save_dir_all)
            
            # Crop helmet images
            crop_helmet_images(r, frame, save_dir_helmet)
            
            print(f"Processed image: {filename}")
        else:
            print(f"Skipping non-image file: {filename}")

if __name__ == '__main__':
    path = 'IOCL_best.pt'
    save_dir_all = r"C:\Users\hiten\Desktop\intern\cropimage"
    save_dir_helmet = r"C:\Users\hiten\Desktop\intern\cropimage\helmet"
    model = load_model(path)
   
    # Create the save directories if they don't exist
    os.makedirs(save_dir_all, exist_ok=True)
    os.makedirs(save_dir_helmet, exist_ok=True)
    
    
    # Start the violation checking process in a separate thread
    try:
        _thread.start_new_thread(check_violation, (r"C:\\Users\\hiten\\Desktop\\intern\\imageor", save_dir_all, save_dir_helmet,))
    except:
        print("Error starting thread")

    while True:
        pass
