import matplotlib.pyplot as plt
# from easyocr import Reader
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask , render_template,request,jsonify
import os
app = Flask(__name__)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/uploads/')
import re
import requests
from PIL import Image
from io import BytesIO
from PIL import Image
import io
import base64
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
state_codes = { "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TS": "Telangana", "TR": "Tripura", "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal", "AN": "Andaman and Nicobar Islands", "CH": "Chandigarh", "DH": "Dadra and Nagar Haveli", "DD": "Daman and Diu", "DL": "Delhi", "LD": "Lakshadweep", "PY": "Puducherry" }
digit_to_letter_mapping = { '8': 'B', 'G': '6', '0': 'D', '5': 'S' }
def hamming_distance(str1, str2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

def predict_first_letter(second_letter):
    possible_first_letters = [code[0] for code in state_codes.keys() if code[1] == second_letter]
    if possible_first_letters:
        return possible_first_letters[0]
    return 'T'

def correct_state_code(number_plate):
    state_code = number_plate[:2]
    if state_code in state_codes:
        return number_plate
    closest_code = min(state_codes.keys(), key=lambda code: hamming_distance(state_code, code))
    if hamming_distance(state_code, closest_code) <= 1:
        corrected_number_plate = closest_code + number_plate[2:]
        return corrected_number_plate
    return number_plate

def correct_and_check_number_plate(number_plate):
    corrected_number_plate = list(number_plate)
    for i in range(min(4, len(corrected_number_plate))):
        if corrected_number_plate[i] in digit_to_letter_mapping:
            corrected_number_plate[i] = digit_to_letter_mapping[corrected_number_plate[i]]
    corrected_number_plate = ''.join(corrected_number_plate)
    if len(corrected_number_plate) < 10:
        if len(corrected_number_plate) == 9 and corrected_number_plate[0].isalpha():
            first_letter = predict_first_letter(corrected_number_plate[0])
            corrected_number_plate = first_letter + corrected_number_plate
        elif len(corrected_number_plate) == 9:
            corrected_number_plate = corrected_number_plate[:1] + '0' + corrected_number_plate[1:]
    corrected_number_plate = correct_state_code(corrected_number_plate)
    return corrected_number_plate
@app.route('/')
def home():
    return render_template('img.html')
@app.route('/number_plate',methods = ['POST','GET'])
def License_detection():
# Match contours to license plate or character template
    def find_contours(dimensions, img) :

    # Find all contours in the image
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Retrieve potential dimensions
        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]

        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

        ii = cv2.imread('contour.jpg')

        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs :
            #detects contour in binary image and returns the coordinates of rectangle enclosing it
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

            #checking the dimensions of the contour to filter out the characters by contour's size
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

                char_copy = np.zeros((44,24))
                #extracting each character using the enclosing rectangle's coordinates.
                char = img[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))

                cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
                plt.imshow(ii, cmap='gray')

    #             Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy) #List that stores the character's binary image (unsorted)

        #Return characters on ascending order with respect to the x-coordinate (most-left character first)

        plt.show()
        #arbitrary function that stores sorted list of character indeces
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])# stores character images according to their index
        img_res = np.array(img_res_copy)

        return img_res
    def croped_plate(image):
        number_plate_model = YOLO(r"C:\Users\yelar\Desktop\purview\nice-main\models\best_nplate.pt")
        img = cv2.imread(image) 
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
             # print(person_on_bike_results)
        number_plate_results = number_plate_model.predict(img_rgb,save = False, name=image)
                # print("number plate results here")
                # print(number_plate_results)
        for np_box in number_plate_results[0].boxes:
                isfound=False
                np_cls = np_box.cls
                # print(number_plate_model.names[int(np_cls)])
                np_x1,np_y1,np_x2,np_y2 = np_box.xyxy[0]
                np_0x1,np_0y1,np_0x2,np_0y2=np_x1,np_y1,np_x2,np_y2
                # Ensure the expanded box does not go out of image bounds
                np_x1 = max(int(np_x1-250), 0)
                np_y1 = max(int(np_y1-250), 0)
                np_x2 = min(int(np_x2+250), img_rgb.shape[1])
                np_y2 = min(int(np_y2+250), img_rgb.shape[0])

                # Extract the adjusted number plate image
                # number_plate_image = img_rgb[np_y1:np_y2, np_x1:np_x2]

                number_plate_image = img_rgb[int(np_y1):int(np_y2),int(np_x1):int(np_x2)]
                if number_plate_image is not None and number_plate_image.size != 0:
                    number_plate_image = cv2.cvtColor(number_plate_image,cv2.COLOR_RGB2BGR)
                # print(np_box.xyxy[0])
                # image_url = image
                image = number_plate_image
                # prepare decoder inputs
                task_prompt = "<s_cord-v2>"
                decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

                pixel_values = processor(image, return_tensors="pt").pixel_values

                outputs = model.generate(
                    pixel_values.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    max_length=model.decoder.config.max_position_embeddings,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
                sequence = processor.batch_decode(outputs.sequences)[0]
                sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                # Replace the specified pattern with an empty string
                print(sequence)
                cleaned_string = re.sub(r'<[^>]+>', '', sequence)
                cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', cleaned_string)
                if cleaned_string.startswith('IND'):
                    cleaned_string=cleaned_string[3:]
                if cleaned_string:
                    print("incoming",sequence,cleaned_string)
                    print(cleaned_string[:10])
                
                    text=cleaned_string[:10]
                    # text=correct_and_check_number_plate(text)
                    '''
                    
                    
                    WRITE THE CRIMINAL DATABASE LOGIC FOR SEARCHING STOLEN VEHCILES
                    
                    
                    
                    '''
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5  # Increase the font scale conservatively to make the text slightly larger
                      # Red color (BGR)
                    text_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
                    text_x = int(np_0x1 + (np_0x2 - np_0x1 - text_size[0]) / 2)
                    text_y = int(np_0y1 - 10)  # Adjust vertical position
                    if isfound:
                        color = (0, 0, 255)
                        text_color = (0, 0, 255)
                    else: 
                        color = (0, 255, 0)  # Green color (BGR)
                        text_color = (0, 255, 0)
                    # print(text_color,"&&&&&&&&&")
                    thickness = 3  # Thickness of the bounding box lines
                    cv2.rectangle(img, (int(np_0x1), int(np_0y1)), (int(np_0x2), int(np_0y2)), color, thickness)
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, text_thickness)
                    cv2.imwrite("contour.jpg", img)
        image_file =r"C:\Users\yelar\Desktop\purview\nice-main\contour.jpg"
        image = Image.open(image_file)
        # Simulate processing and create a response image
        processed_image = image  # In a real scenario, process the image

        # Convert processed image to byte array
        byte_arr = io.BytesIO()
        processed_image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()

        # Encode image to Base64
        encoded_image = base64.b64encode(byte_arr).decode('utf-8')
        response = {
            "description": text,
            "image":encoded_image
        }
        return response
        # Print the cleaned string
    # Find characters in the resulting images
# Replace 'path/to/image.jpg' with your image path
    img = request.form['text']
    print(img)
    output=croped_plate(img)
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)
