import cv2
import numpy as np
from bloodcellcount import app
import os

class blood_cell_count:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    #loading weights in construnctor
    def __init__(self):
        weights_path = os.path.join(app.root_path, 'static/saved_models', 'yolov3_final.weights')
        config_path = os.path.join(app.root_path, 'static/cfg', 'yolov3.cfg')
        self.net = cv2.dnn.readNet(weights_path, config_path)
    
    #getting output layer of the net
    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers    

    #drawing boxes on image 
    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS, classes):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #detecting cells using loaded net, counting cells and drawing boxes on img
    def cell_detect(self, image_file):
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        scale = 0.00392
        class_ids = []
        Width = image_file.shape[1]
        Height = image_file.shape[0]
        #coverting numpy array to blob
        blob = cv2.dnn.blobFromImage(image_file, scale, (608,608), (0,0,0), True, crop=False)
        #prediction
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(self.net))
        #initializing counter, color and class name for rbc, wbc, platelates respectively  
        count_list = [0 ,0 ,0]
        COLORS = [(255, 0, 0),(0, 0, 0),(0, 0, 255)]
        classes = ['rbc', 'wbc', 'platelates']
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])                        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for index in indices:
            box_index = index[0]
            box = boxes[box_index]
            
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image_file, class_ids[box_index], confidences[box_index], round(x), round(y), round(x+w), round(y+h), COLORS, classes)
            count_list[class_ids[box_index]] = count_list[class_ids[box_index]]+1 
        result = np.array([count_list[0], count_list[1], count_list[2]])
        return result, image_file
