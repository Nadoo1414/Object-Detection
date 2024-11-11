#Road Turn Detection with YOLOv8
##Project Summary
The "Road Turn Detection with YOLOv8" project focuses on developing a computer vision model that can detect road turns in real-time. By using the YOLOv8 model, this system is trained to identify road bends and turns from a custom dataset of road images. This application is designed to support navigation and enhance road safety through accurate turn detection.

Table of Contents
Project Summary
Project Workflow
Key Highlights
Requirements
Installation and Setup
Usage
Future Work
Contact
##Project Workflow
1. Dataset Collection and Annotation
Collected a custom dataset of road images, covering diverse turn scenarios.
Annotated each image with bounding boxes around turn regions and labeled them as "Turn."
2. Model Configuration
Configured a YOLOv8 file to define the model architecture, anchor boxes, and hyperparameters optimized for road turn detection.
3. Darknet Framework Setup
Set up the Darknet framework for YOLO model training by downloading, configuring, and building it.
4. Data Configuration File
Created a configuration file to link the custom dataset with the YOLOv8 model configuration.
5. Pre-trained Weights
Initialized the YOLOv8 model with pre-trained weights to improve accuracy and reduce training time.
6. Model Training
Trained the YOLOv8 model on the custom dataset, adjusting hyperparameters for optimal detection accuracy.
Fine-tuned to achieve high precision in road turn detection.
7. Evaluation
Evaluated the model using metrics like mean Average Precision (mAP) on a validation dataset to measure detection accuracy.
8. Real-Time Inference
The YOLOv8 model was deployed for real-time inference, effectively detecting turns in road images to aid navigation and road safety.
Key Highlights
Custom Dataset: Tailored with diverse turn and angle scenarios to improve model robustness.
YOLOv8 Model: Configured specifically for detecting road turns using the Darknet framework.
High Precision: Achieved high detection accuracy through model fine-tuning and validation.
Real-Time Inference: Model is optimized for real-time deployment to assist with navigation.
Requirements
Framework: Darknet
Programming Language: Python
Libraries: OpenCV, NumPy
Additional: Pre-trained YOLOv8 weights
Installation and Setup
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/road-turn-detection-yolov8.git
Install necessary libraries:
bash
Copy code
pip install -r requirements.txt
Set up the Darknet framework following the installation guide here.
Download the pre-trained YOLOv8 weights and place them in the appropriate directory.
Usage
Prepare your dataset following the annotated format required by YOLO.
Train the model using the command:
bash
Copy code
./darknet detector train <data_config> <yolov8_config> <pre-trained_weights>
To run real-time inference on images, use:
bash
Copy code
./darknet detector test <data_config> <weights_file> <image_file>
Future Work
Expand the dataset to include varying weather conditions, lighting, and road types.
Experiment with different object detection models to improve accuracy and efficiency.
Optimize the model for low-latency applications.
Contact
For any questions or issues, feel free to reach out to Nadia Ali at nadiaalee786@gmail.com.
