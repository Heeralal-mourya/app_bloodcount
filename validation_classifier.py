from tensorflow.keras.models import load_model
import numpy as np
import os, cv2

class validation_model():
    model = None
    labels = ['negative', 'positive']
    
    def __init__(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_path,"static/saved_models","blood_cell_256_validation_mobilenet_3.h5")
        self.model = load_model(model_path)
        return None    

    def predict(self, img):
        pred = self.labels[np.argmax(self.model.predict(img),axis=1)[0]]
        return pred