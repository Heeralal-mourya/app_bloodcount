import os
import cv2
from flask import current_app
import cv2

#saving image in static/Uploaded
def save_image(cur_img, picture_name):
    cur_img.save(os.path.join(current_app.root_path, 'static/Uploaded', picture_name))
    return picture_name

#To load image from static/Uploaded with required dimensions
def get_img(img_path):
    img_classifier = cv2.imread(img_path)
    img_validation =  cv2.resize(img_classifier, (256, 256))
    img_validation = img_validation.reshape((1,256,256,3))
    return img_classifier, img_validation
    
