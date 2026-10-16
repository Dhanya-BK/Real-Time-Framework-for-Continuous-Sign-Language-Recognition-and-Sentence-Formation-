 Real-Time Framework for Continuous Sign Language Recognition and Sentence Formation
--------------------------------------------------------------------------------------
 Overview
-------------
This project presents an AI-based framework for recognizing continuous sign language gestures in real time and converting them into meaningful sentences. Unlike traditional sign language systems that recognize only isolated gestures, this framework captures temporal information using Long Short-Term Memory (LSTM) networks, enabling continuous gesture recognition and sentence formation.


The system uses MediaPipe  for landmark extraction, TensorFlow for sequence learning, and OpenCV for real-time video processing.




Features
------------
- Real-time sign language recognition using webcam
- Continuous gesture recognition using LSTM
- MediaPipe Holistic landmark extraction
- Automatic sentence formation
- High recognition accuracy using temporal sequence modeling
- Low latency real-time prediction
- Modular and scalable architecture



Technologies Used
----------------------
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe Holistic
- NumPy
- Joblib
- Pickle


System Workflow
------------------
1. Capture live video using webcam.
2. Detect body, face, and hand landmarks using MediaPipe Holistic.
3. Extract normalized landmark coordinates.
4. Generate sequences of landmark frames.
5. Feed sequences into the trained LSTM model.
6. Predict sign language gestures.
7. Form meaningful sentences from recognized gestures.
8. Display the generated sentence in real time.



 Model Architecture
-----------------------
- Input: Landmark sequences extracted using MediaPipe Holistic
- Sequence Length: 20 Frames
- Model: Long Short-Term Memory (LSTM)
- Output: Gesture Classification
- Sentence Formation using sequential prediction logic

 Dataset
-----------
The dataset consists of multiple sign language gestures recorded under different conditions. Each gesture is converted into landmark sequences before training.

Future Enhancements
------------------------
- Transformer-based sequence modeling
- Speech synthesis for generated sentences
- Multi-language support
- Mobile application deployment


Applications
------------
- Communication assistance for the deaf and hard-of-hearing community
- Educational institutions
- Public service accessibility
- Healthcare communication
- Smart assistive technologies

Authors
------------
Akshaya Shankari B
Dhanya B K
Sharanya Rai K
Srijanya S R
