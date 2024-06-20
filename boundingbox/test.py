import numpy as np
import cv2
from ultralytics import YOLO
def croped_plate(image):
        number_plate_model = YOLO(r"C:\Users\yelar\Desktop\license_plate\flask_now\static\models\best_nplate.pt")
        img = cv2.imread(image_path) 
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
             # print(person_on_bike_results)
        number_plate_results = number_plate_model.predict(img_rgb,save = False, name=image)
                # print("number plate results here")
                # print(number_plate_results)
        for np_box in number_plate_results[0].boxes:
                np_cls = np_box.cls
                # print(number_plate_model.names[int(np_cls)])
                np_x1,np_y1,np_x2,np_y2 = np_box.xyxy[0]
                number_plate_image = img_rgb[int(np_y1):int(np_y2),int(np_x1):int(np_x2)]
                number_plate_image = cv2.cvtColor(number_plate_image,cv2.COLOR_RGB2BGR) 
                color = (0, 255, 0)  # Green color (BGR)
                thickness = 3  # Thickness of the bounding box lines
                # print(np_box.xyxy[0])
                cv2.rectangle(img, (int(np_x1), int(np_y1)), (int(np_x2), int(np_y2)), color, thickness)
                cv2.imwrite("contour.jpg", img)
image_path=r"C:\Users\yelar\Desktop\license_plate\flask_now\static\uploads\cover_3.jpeg"
croped_plate(image_path)