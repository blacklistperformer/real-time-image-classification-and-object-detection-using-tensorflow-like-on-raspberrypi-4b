# real-time-image-classification-and-object-detection-using-tensorflow-like-on-raspberrypi-4b
BTech Seventh semester Project where we have done real time image classification and object detection using tensorflow like on raspberrypi 4b for surveillance purpose

---
# Real-Time Image Classification and Object Detection on Raspberry Pi 4

Welcome to the Real-Time Image Classification and Object Detection project using TensorFlow on Raspberry Pi 4! This repository provides code for running real-time image classification and object detection on a Raspberry Pi 4 using TensorFlow Lite.

## Prerequisites

Before running the code, ensure that your Raspberry Pi is up to date. Open a terminal and execute the following commands:

```bash
sudo apt update
sudo apt upgrade
```

## Download and Setup

 - Clone this repository to your Raspberry Pi:
```bash
git clone https://github.com/blacklistperformer/real-time-image-classification-and-object-detection-using-tensorflow-like-on-raspberrypi-4b.git
```
 - Move the repository to a directory of your choice and rename it:
 ```bash
sudo mv real-time-image-classification-and-object-detection-using-tensorflow-like-on-raspberrypi-4b tflite1

```

 - Install virtualenv and create a virtual environment:
 ```bash
sudo pip3 install virtualenv
python3 -m venv tflite1-env
source tflite1-env/bin/activate
```

 - Run the script to install required dependencies:
 ```bash
bash tflite1/get_pi_requirements.sh
```

## Run the Simulation

 - Once you've completed the setup, you can run the simulation using the following command:
```bash
python3 tflite1/TFLite_detection_webcam.py --modeldir=Sample_TFLite_model
```

This command launches the real-time image classification and object detection using the specified TensorFlow Lite model. Adjust the --modeldir parameter to point to the directory containing your desired TFLite model.

Feel free to explore and modify the code to suit your needs. If you encounter any issues or have questions, check the repository for updates or open an issue.

Happy coding! 🚀

------------------------------------------------------
>This code is written by Neeraj Nikhil Roy 20/11/EC/053 as a part of Project for BTech as a student of
School of Enginnering, Jawaharlal Nehru University, New Delhi
The guide for this project is Dr. Ankit Kumar Jaiswal
This code is written for the purpose of Object Detection using Tensorflow Lite and OpenCV.
The code is written for the purpose of detecting objects in real time using webcam.
Title of the project is: REAL TIME IMAFE CLASSIFICATION AND OBJECT DETECTION USING TENSORFLOW LITE ON RASPBERRYPI 4B
Members in my team are as follow
Kshama Meena 20/11/EC/055 kshamameena7@gail.com
Komal Kesav Nenavath 20/11/EC/012
Divyansh Singh 20/11/EC/057
Github link to the repository https://github.com/blacklistperformer/real-time-image-classification-and-object-detection-using-tensorflow-like-on-raspberrypi-4b
Link to my other socials
Instagram: https://www.instagram.com/blacklistperformer/
Linkedln: https://www.linkedin.com/in/neeraj-roy-556968192/
Github: https://github.com/BlackListPerformer
Stackoverflow: https://stackoverflow.com/users/19916561/neeraj-roy
Email: neerajroy06502@gmail.com
>Kshama Meena's social
Linkedln: https://www.linkedin.com/in/kshama-meena-1851a8207/
Github: https://github.com/kshamameena
>Komal's Social: Github: https://github.com/komalkesav

------------------------------------------------------
