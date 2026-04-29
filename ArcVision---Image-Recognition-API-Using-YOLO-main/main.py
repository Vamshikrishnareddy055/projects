# Import necessary libraries
from flask import Flask, render_template, request
import cv2
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# Define paths for model weights, configuration, and classes
script_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(script_path, 'tiny.weights')
cfg_path = os.path.join(script_path, 'tiny.cfg')
classes_path = os.path.join(script_path, 'classes.txt')

# Load pre-trained neural network weights and configuration
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load classes from a file
classes = []
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route for uploading images
@app.route('/upload', methods=['POST'])
def upload():
    # Get uploaded image file
    img = request.files['image']
    img.save(os.path.join(script_path, 'static/input.jpg'))

    # Read uploaded image
    image = cv2.imread(os.path.join(script_path, 'static/input.jpg'))
    height, width, _ = image.shape

    # Preprocess the image for neural network input
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass through the neural network
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store detected object information
    class_ids = []
    confidences = []
    boxes = []

    # Process the output of the neural network
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 1, color, 2)

    # Save the output image
    cv2.imwrite(os.path.join(script_path, 'static/output.jpg'), image)

    # Render the result template with input and output image paths
    return render_template('result.html', input_image='static/input.jpg', output_image='static/output.jpg')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
