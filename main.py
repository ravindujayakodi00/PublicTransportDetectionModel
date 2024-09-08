from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import pickle
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load YOLOv8 model (custom trained model for bus boards)
model = YOLO('model/yolov8_custom_trained.pt')

# Load the OCR route dictionary
model_dir = 'model'
with open(os.path.join(model_dir, 'route_dict.pkl'), 'rb') as f:
    route_dict = pickle.load(f)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])


def detect_bus_board(frame):
    # Detect objects using YOLO
    results = model.predict(frame, conf=0.5)
    bus_board_box = None

    # Loop through detections to find the bus_board
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, box)

        # Get the object class (label)
        class_id = int(result.cls[0].item())
        object_type = model.names[class_id]  # Get the object type name
        # If the detected object is the bus board, return the bounding box
        if object_type == "bus_board,":
            bus_board_box = (x1, y1, x2, y2)
            break  # Only process the first bus_board detected

    return bus_board_box


def recognize_route_number(image):
    # Use EasyOCR to detect bus route number
    result = reader.readtext(image, detail=0, allowlist='0123456789')
    if result:
        recognized_number = ''.join(result).lstrip('0')  # Remove leading zeros
        return recognized_number
    return None


def get_route_name(route_number):
    # Get route name from route number
    return route_dict.get(route_number, "Route not found")


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Load the image
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Step 1: Detect bus_board using YOLO
    bus_board_box = detect_bus_board(img)

    if bus_board_box is not None:
        # Step 2: Crop the image to the bus_board area
        x1, y1, x2, y2 = bus_board_box
        cropped_board = img[y1:y2, x1:x2]

        # Save the cropped bus board image
        cropped_board_path = 'cropped_images/cropped_bus_board.jpg'
        cv2.imwrite(cropped_board_path, cropped_board)

        # Step 3: Further crop to the middle part of the bus_board where the number is located (optional)
        h, w, _ = cropped_board.shape
        middle_x1 = int(w * 0.4)
        middle_x2 = int(w * 0.6)
        cropped_number_area = cropped_board[:, middle_x1:middle_x2]

        # Save the cropped number area image
        cropped_number_path = 'cropped_images/cropped_number_area.jpg'
        cv2.imwrite(cropped_number_path, cropped_number_area)

        # Step 4: Use OCR to detect the bus route number
        recognized_number = recognize_route_number(cropped_number_area)

        if recognized_number:
            # Step 5: Get the route name from the recognized number
            route_name = get_route_name(recognized_number)
            print(f"Recognized route number: {recognized_number} - Route name: {route_name}")
            return jsonify({
                "route_number": recognized_number,
                "route_name": route_name,
                "cropped_bus_board_path": cropped_board_path,
                "cropped_number_path": cropped_number_path
            }), 200
        else:
            # Handle case where OCR failed to detect the number
            return jsonify({
                "error": "Could not recognize the bus route number",
                "cropped_bus_board_path": cropped_board_path,
                "cropped_number_path": cropped_number_path
            }), 400
    else:
        # Handle case where no bus board was detected
        return jsonify({"error": "No bus board detected"}), 400


if __name__ == '__main__':
    # Ensure the directory exists for saving cropped images
    os.makedirs('cropped_images', exist_ok=True)
    app.run(host='0.0.0.0', port=9090, debug=True)
