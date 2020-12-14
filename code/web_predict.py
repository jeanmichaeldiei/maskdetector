import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import cv2
import imutils
import pathlib
from imutils.video import VideoStream

import sys
from object_detection import ObjectDetection

MODEL_FILENAME = '../model/model.tflite'
LABELS_FILENAME = '../model/labels.txt'
CONFIDENCE = 0.4

class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]

    def annotate_frame(self, predictions,w,h,open_cv_image):
        flag = 0
        for pred in predictions:
            prob = pred['probability']
            if  prob > CONFIDENCE:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                pred_box = pred['boundingBox']

                startY = int(pred_box['top'] * h)
                startX = int(pred_box['left'] * w)

                endY = int(((pred_box['top']+pred_box['height'])) * h)
                endX = int(((pred_box['left']+pred_box['width'])) * w)
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                label = "Mask" if pred['tagName'] == 'has mask' else "No Mask"
                color = (0, 255, 0) if pred['tagName'] == 'has mask' else (0, 0, 255)
                
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, prob * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(open_cv_image, label, (startX+5, startY - 10),
                cv2.FONT_ITALIC, 0.50, color, 2)
                cv2.rectangle(open_cv_image, (startX, startY), (endX, endY), color, 2)
                flag = 1

        return open_cv_image, flag

class MyVideoStream():
    def __init__(self, model):
        self.model = model

        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        self.running_vs = cv2.VideoCapture(0)
        #self.isRunning = False
        self.do_preds = False
    def predict_frame(self):
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 500 pixels
        if self.running_vs:
            ret, frame = self.running_vs.read()
            if frame is None:
                return ''.encode()
            orig_h, orig_w = frame.shape[:2]
            new_w, new_h = 500, int(500*orig_h/orig_w)
            frame = cv2.resize(frame, (new_w, new_h))
            
            if self.do_preds:
                predictions = self.model.predict_image(Image.fromarray(frame))
                frame, flag = self.model.annotate_frame(predictions,new_w,new_h,frame)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            print("Error getting webcam frames.")
            return ''.encode()

def create_model():
    #Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    return TFLiteObjectDetection(MODEL_FILENAME, labels)

def predict_image(model, image_filename):
    image = Image.open(image_filename)
    open_cv_image = cv2.imread(str(image_filename),1)
    h,w = open_cv_image.shape[:2]

    predictions = model.predict_image(image)
    frame, flag = model.annotate_frame(predictions,w,h,open_cv_image)
    return frame, flag
