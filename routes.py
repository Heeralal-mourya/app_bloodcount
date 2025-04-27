from flask import Flask, render_template, url_for, redirect, request, jsonify, Response 
from bloodcellcount import app
from bloodcellcount import validation_classifier
from bloodcellcount import blood_cell_detection
from bloodcellcount.util import save_image,  get_img
import numpy as np
import os
import cv2
import time

model_validation = validation_classifier.validation_model()
model_obj = blood_cell_detection.blood_cell_count()

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('blood-count.html')

@app.route("/diag", methods=['GET', 'POST'])
def diag():
    
    #loading image and saving
    try:
        temp_id = request.form.get('temp_id')
        timestamp = str(time.time()).split(".")[0]
        img_file = request.files.get('image_file')
        file_name = temp_id + '_' + timestamp + '_' + img_file.filename
        img_name = save_image(img_file, file_name)
        img_path = os.path.join(app.root_path, 'static/Uploaded', img_name)
        img_classifier, img_validation = get_img(img_path)
    except Exception as msg:
        print("[ERROR] in getting blood cell image at : " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp    
    
    #image validation check
    try:
        result_validation = model_validation.predict(img_validation)
        if(result_validation=='negative'):
            raise Exception('Invalid Image')
    except Exception as msg:
        print("[ERROR] in validating image : " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    
    #prediction for cell detection and counts
    try:
        result, image = model_obj.cell_detect(img_classifier)
        result_path = os.path.join(app.root_path, 'static/result', file_name)
        cv2.imwrite(result_path,image)
        ##upload result image on s3 bucket and send s3_url in response
    except Exception as msg:
        print("[ERROR] in detecting blood cell counts: " + time.asctime(time.localtime(time.time())) + " : " + str(msg))
        data = {'status' : False, 'message' : str(msg)}
        resp = jsonify(data)
        resp.status_code = 200
        return resp
    
    dict = {'rbc': int(result[0]), 'wbc':int(result[1]), 'platelates': int(result[2]), 'status': True}
    return (dict)