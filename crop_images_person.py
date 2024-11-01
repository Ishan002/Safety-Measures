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

def crop_person_image(boxes, frame, save_dir):
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


def custom(path_or_model='path/to/model.pt', autoshape=True):
    """custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cuda')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model
    model.conf = 0.60  # NMS confidence threshold
    model.iou = 0.45 
    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)

def get_center(cord):
    cent_x = (cord[0] + cord[2])/2
    cent_y = (cord[1] + cord[3])/2
    return cent_x,cent_y

def get_polygon_status(boxes):
    person_poly = False
    spark_poly = False
    #for m in range(np.count_nonzero(boxes.cls == 2)):
    for m in range(len(boxes.data)):
        label = boxes.data[m][5]
        if label == 2:
            cent_x,cent_y=get_center(boxes.data[m])
            print(cent_x,cent_y)
    return person_poly,spark_poly

def image_colorfulness( image, satadj):
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype("float32")
    
        (h, s, v) = cv.split(image)
        s = s*satadj
        s = np.clip(s,0,255)
        imghsv = cv.merge([h,s,v])
        imgrgb = cv.cvtColor(imghsv.astype("uint8"), cv.COLOR_HSV2BGR)
        return imgrgb

def draw_boxes(results, img, classes):
    if results.numel() > 0:  # Check if results tensor is not empty
        for pred in results:
            if pred.numel() > 0:  # Check if pred tensor is not empty
                x, y, w, h = pred[:4]  # Assuming pred contains coordinates directly
                label_class = int(pred[5])
                # Draw bounding box
                cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), thickness=2)
                # Draw class label
                cv2.putText(img, classes[label_class], (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 1)
    return img




def load_model(ckpt_path, conf=0.10, iou=0.25):
    model = torch.hub.load('','custom',
                           path_or_model=ckpt_path,source='local',force_reload=True)
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
     # maximum number of detections per image
    return model

def image_preprocess(frame, h, w):
    # Resize the image and convert it to RGB if it's not already in RGB format
    frame_resized = cv.resize(frame, (w, h))
    if len(frame_resized.shape) == 2:  # If the image is grayscale, convert it to RGB
        frame_resized = cv.cvtColor(frame_resized, cv.COLOR_GRAY2RGB)
    
    # Convert the image to float32
    frame_resized = frame_resized.astype(np.float32)
    
    # Normalize the image
    frame_resized /= 255.0
    
    # Convert the image to tensor
    img = torch.from_numpy(frame_resized).unsqueeze(0).permute(0, 3, 1, 2).to('cuda')
    
    return img



def get_count_violation(boxes):
    no_helmet_count=0
    no_jumpsuit_count = 0 
    for m in range(len(boxes.data)):
        results = ((boxes.data[m].cpu()).numpy())
        label = boxes.data[m][5]
        if label == 1:
            no_helmet_count+=1
            k+=1
        if label == 3:
            no_jumpsuit_count+=1
            p+=1
    return no_helmet_count,no_jumpsuit_count

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=2,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1,lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness,lineType=cv2.LINE_AA)
    return text_size

def check_violation(image_folder, save_dir):
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
            
            # Draw boxes and crop person images
            frame_with_boxes = frame.copy()
            frame_with_boxes = draw_boxes(r, frame_with_boxes, model.names)
            crop_person_image(r, frame, save_dir)
            
            # Display or save annotated image
            annotated_img_path = os.path.join(save_dir, f"annotated_{filename}")
            cv.imwrite(annotated_img_path, frame_with_boxes)
            
            print(f"Processed image: {filename}")
        else:
            print(f"Skipping non-image file: {filename}")


if __name__ == '__main__':
    k=0
    p=0
    path='IOCL_best.pt'
    model = load_model(path)
    classes = ["helmet", "nohelmet", "jumpsuit", "nojumpsuit", "person"]
    try:
        _thread.start_new_thread(check_violation, (r"C:\\Users\\hiten\\Desktop\\intern\\imageor",r"C:\\Users\\hiten\\Desktop\\intern\\cropimage\\img",))
    except:
        print("error")

    while True:
        pass
