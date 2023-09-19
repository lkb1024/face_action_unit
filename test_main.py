import cv2
import numpy as np
from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity
from typing import Dict
import operator
import torchvision
import torch
from facetorch.datastruct import ImageData
from PIL import ImageFont, ImageDraw, Image

from fau_inference import fau_inference, initialize_fau_model

import mediapipe as mp

import math
from typing import List, Mapping, Optional, Tuple, Union

# For webcam input:
cap = cv2.VideoCapture(0)

labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
model_path = "/Users/felix/Documents/Workspace/mediapipeDemos/models/efficient_face_model.tflite"

path_img_input="./test.jpg"
path_img_output="/test_output.jpg"
path_config="notebooks/gpu.config.yml"

FONTS =cv2.FONT_HERSHEY_SIMPLEX
# FONTS = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf")

mp_face_detection = mp.solutions.face_detection
face_detection =  mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)
fau_net = initialize_fau_model()

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background 
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img

def extract_faces(raw_frame, results, x_scale=1.0, y_scale=1.0):
    frames = []
    bboxes = []
    if results.detections is None:
        return None, None
    for detection in results.detections:
        image_size = raw_frame.shape[1::-1]
        x_min = detection.location_data.relative_bounding_box.xmin
        y_min = detection.location_data.relative_bounding_box.ymin
        width = detection.location_data.relative_bounding_box.width
        height = detection.location_data.relative_bounding_box.height

        x_min = image_size[0] * x_min
        y_min = image_size[1] * y_min
        width = image_size[0] * width
        height = image_size[1] * height
        x_max = x_min + width
        y_max = y_min + height

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        width = x_scale * width
        height = y_scale * height

        x_min = x_center - width / 2
        y_min = y_center - height / 2

        x_max = x_min + width
        y_max = y_min + height

        x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

        frame = raw_frame[y_min:y_max, x_min:x_max]
        bbox = torch.tensor([[x_min, y_min, x_max,y_max]])

        if frame.any():
            frames.append(frame)
            bboxes.append(bbox)

    return frames, bboxes

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    aus_list = []

    if results.detections:    
        face_frames, bboxes = extract_faces(image, results)
        torch_image =torch.from_numpy(np.array(image).transpose((2, 0, 1))).contiguous()

        infostr_probs,  infostr_aus = fau_inference(fau_net, face_frames[0])
        aus_list = list(infostr_aus)

        # # Draw the face detection annotations on the image.  
        torch_image = torchvision.utils.draw_bounding_boxes(torch_image, bboxes[0], labels=['0'], colors='green',
       width=3)
        pil_image = torchvision.transforms.functional.to_pil_image(torch_image)
        
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    image = cv2.flip(image, 1)

    for i, label in enumerate(aus_list):
        colorBackgroundText(image,  label, FONTS, 0.5, (10, 100 + i*30), 1, (0, 255, 0))
    
    cv2.imshow('MediaPipe Face Mesh', image)
        
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()