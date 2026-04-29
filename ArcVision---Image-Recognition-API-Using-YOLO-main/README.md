# ArcVision - Image Recognition API

![Login](/Images/Login.png)

## Introduction

ArcVision is a Flask-based Image Recognition API designed to simplify image recognition tasks using a pre-trained machine learning model fine-tuned for smaller devices, specifically YOLO Lite (You Only Look Once). Users can upload images to receive predictions or information about the recognized objects.

## Features

- **User-friendly Interface:** The API provides a simple web interface for easy image uploading.
- **Object Detection:** Utilizes the YOLO Lite model for efficient object detection in images.
- **Output Visualization:** Recognized objects are highlighted in the output image with bounding boxes and labels.

## Screenshots

### Login
![Login](/Images/Login.png)

### Index
![Index](/Images/Index.png)

### Input
![Input](/Images/Input.jpeg)

### Output
![Output](/Images/Output.png)

## Getting Started

### Prerequisites

Ensure you have a computer or server with the following installed:

- Python
- Flask
- OpenCV

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/KOTAHARSHA25/ArcVision---Image-Recognition-API.git
   ```

2. **Change Directory:**

   ```bash
   cd ArcVision
   ```

3. **Create and Activate Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

4. **Install the Requirements:**

   ```bash
   pip install flask opencv-contrib-python
   ```

5. **Run the Application:**

   ```bash
   python app.py
   ```

   Or, if you are using PythonAnywhere:

   ```bash
   python main.py
   ```

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Upload an image using the provided form.
3. Click on the "Detect Object" button.
4. View the results on the result page, highlighting the recognized objects in the output image.

## Project Structure

- `app.py`: The main Flask application file containing the routes and image processing logic.
- `templates/`: Folder containing HTML templates for the web interface.
  - `templates/index.html`: Main page for image upload.
  - `templates/result.html`: Page displaying input and output images with recognized objects.
