import cv2
import numpy as np
import time
import os
import tempfile
import pyttsx3
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


net = cv2.dnn.readNet("C:\\Users\\lenovo\\Desktop\\sd\\yolov3.weights", "C:\\Users\\lenovo\\Desktop\\sd\\yolov3.cfg")


with open("C:\\Users\\lenovo\\Desktop\\sd\\coco.names", 'r') as f:
    classes = [line.strip() for line in f]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


engine = pyttsx3.init()
layers_names=net.getLayerNames()

cap = cv2.VideoCapture(0) 
temp_dir = tempfile.mkdtemp()

def get_clothing_color(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 50, 50), (20, 255, 255))  
        color = "red" if cv2.countNonZero(mask) > 0 else "unknown" 
    except Exception as e:
        print(f"Error in color detection: {e}")
        color = "unknown"
    return color

def generate_description(detected_objects, person_boxes, frame):
    if not detected_objects:
        return ["No objects detected."]

    descriptions = []

    for i, box in enumerate(person_boxes):
        if i < len(detected_objects):
            label_id = detected_objects[i]
            if label_id < len(classes):
                label = classes[label_id]
                clothing_color = get_clothing_color(frame, box)
                direction = "unknown"  # Placeholder for direction detection (to be implemented)
                
                description = f"I see a {label} wearing {clothing_color} clothes, located {direction}."
                descriptions.append(description)
            else:
                print(f"Warning: Index {label_id} out of range for 'classes' list.")
        else:
            print(f"Warning: Index {i} out of range for 'detected_objects' list.")

    return descriptions

try:
    while True:
        start_time = time.time()  # Start time for calculating FPS

        ret, frame = cap.read()  # Read frame from the camera
        if not ret:
            print("Error: Failed to capture image from camera")
            break

        height, width = frame.shape[:2]  # Get dimensions of the frame

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:  # Adjust confidence threshold here
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to avoid overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.1)

        # Display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        detected_objects = []
        person_boxes = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detected_objects.append(class_ids[i])
                person_boxes.append([x, y, w, h])

        # Generate scene description
        descriptions = generate_description(detected_objects, person_boxes, frame)

        
        for description in descriptions:
            engine.setProperty('rate', 150)  # Adjust speech rate
            engine.say(description)
            engine.runAndWait()

        
        cv2.imshow('Object Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27: 
            break
finally:
   
    cap.release()
    cv2.destroyAllWindows()


    try:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error deleting {temp_dir}: {e}")

