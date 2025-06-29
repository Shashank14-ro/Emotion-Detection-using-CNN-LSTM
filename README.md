# Emotion-Detection-using-CNN-LSTM
Emotion detection system using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to recognize emotions from sequences of facial images or videos.

# Emotion Detection using CNN & LSTM 🤖🎭

This project implements an advanced facial emotion recognition system using a combination of Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal sequence analysis. It is ideal for recognizing emotional changes across video frames or image sequences.

Haarcascade.xml is used for object detection; in our project, we use it to detect faces, which are then classified using a CNN & LSTM algorithm.

To improve output accuracy, LSTM is integrated into the existing CNN model. While CNN extracts spatial features from facial images, the LSTM network captures temporal dependencies between frames, enabling robust emotion classification across sequences
---

## 🧠 Key Features

- Emotion recognition from image sequences or short videos
- CNN for extracting spatial facial features
- LSTM for modeling temporal dependencies between frames
- Real-time inference capability (optional)
- Trained on labeled video/frame datasets

---

## 🛠 Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow / Keras
  - OpenCV (video frame extraction)
  - NumPy, Pandas
  - Matplotlib / Seaborn (visualization)

---

## 🗃 Dataset
   -FER2013
```
emotion-detection-cnn-lstm/
├── dataset/ # Dataset of frame sequences
├── models/ # Saved model weights
│ └── cnn_lstm_model.h5
├── notebooks/ # Jupyter notebooks for experimentation
│ └── train_model.ipynb
├── src/ # Source code
│ ├── preprocess.py
│ ├── model.py
│ ├── train.py
│ └── predict.py
```

📈 Model Architecture
CNN: For per-frame spatial feature extraction

LSTM: For analyzing the temporal emotion progression across frames

Dense: Fully connected layer with softmax for final emotion classification

🎯 Emotions Detected
```
Happy

Sad

Angry

Surprise

Fear

Neutral
```
