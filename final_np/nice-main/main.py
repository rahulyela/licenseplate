import os
import re
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request
from easyocr import Reader
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/uploads/')

# Initialize models
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# State code and digit-to-letter mappings
state_codes = { "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TS": "Telangana", "TR": "Tripura", "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal", "AN": "Andaman and Nicobar Islands", "CH": "Chandigarh", "DH": "Dadra and Nagar Haveli", "DD": "Daman and Diu", "DL": "Delhi", "LD": "Lakshadweep", "PY": "Puducherry" }
digit_to_letter_mapping = { '8': 'B', '6': 'G', '0': 'D', '5': 'S' }

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

@app.route('/', methods=['POST', 'GET'])
def license_detection():
    def croped_plate(image_path):
        number_plate_model = YOLO("path_to_your_model/best_nplate.pt")
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        number_plate_results = number_plate_model.predict(img_rgb, save=False, name=image_path)
        for np_box in number_plate_results[0].boxes:
            np_x1, np_y1, np_x2, np_y2 = np_box.xyxy[0]
            np_x1 = max(int(np_x1-250), 0)
            np_y1 = max(int(np_y1-250), 0)
            np_x2 = min(int(np_x2+250), img_rgb.shape[1])
            np_y2 = min(int(np_y2+250), img_rgb.shape[0])
            number_plate_image = img_rgb[int(np_y1):int(np_y2), int(np_x1):int(np_x2)]
            if number_plate_image is not None and number_plate_image.size != 0:
                number_plate_image = cv2.cvtColor(number_plate_image, cv2.COLOR_RGB2BGR)
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            pixel_values = processor(number_plate_image, return_tensors="pt").pixel_values
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
            cleaned_string = re.sub(r'<[^>]+>', '', sequence)
            cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', cleaned_string)
            if cleaned_string.startswith('IND'):
                cleaned_string = cleaned_string[3:]
            if cleaned_string:
                return cleaned_string[:10], img
        return None, None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        file.save(file_path)
        number_plate, annotated_img = croped_plate(file_path)
        if number_plate:
            corrected_number_plate = correct_and_check_number_plate(number_plate)
            cv2.imwrite(os.path.join(UPLOAD_PATH, 'result.jpg'), annotated_img)
            return render_template('result.html', number_plate=corrected_number_plate)
        else:
            return "Number plate not detected"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
