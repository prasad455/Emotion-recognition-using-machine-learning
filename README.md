
#  Emotion Recognition using Machine Learning

This project is focused on building a Machine Learning model to classify human emotions based on facial expressions. It takes facial images as input and predicts the underlying emotion, such as Happy, Sad, Angry, Surprise, etc.

# Project Overview

Emotion recognition is a key application of computer vision and artificial intelligence. In this project, we used a labeled dataset of facial expressions and trained a classifier to identify emotions accurately.

# Dataset

We used the **FER-2013** dataset (Facial Expression Recognition), which includes grayscale facial images categorized into 7 emotion classes:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) or use your custom dataset.

# Technologies Used

- Python 
- NumPy & Pandas (Data manipulation)
- OpenCV (Image preprocessing and face detection)
- Scikit-learn (Model building and evaluation)
- TensorFlow / Keras (For deep learning model, if used)
- Matplotlib / Seaborn (Visualization)

# Features

- Image preprocessing (grayscale, resizing, normalization)
- Face detection using Haar Cascades 
- Emotion classification using Machine Learning / CNN
- Model evaluation using accuracy, confusion matrix, and precision-recall
- GUI or real-time emotion detection

# Model Performance

| Model          | Accuracy |
|----------------|----------|
| CNN               | 72%      |

# How to Run

1. Clone this repository:
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python train.py
```

4. Run real-time emotion detection (if applicable):
```bash
python detect_emotion.py
```


# Project Structure

```
emotion-recognition/
├── data/                  # Dataset directory
├── models/                # Saved model files
├── notebooks/             # Jupyter notebooks
├── train.py               # Training script
├── detect_emotion.py      # Emotion detection from webcam
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

# Future Enhancements

- Add real-time webcam support for emotion detection
- Deploy as a web app using Flask or Streamlit
- Improve accuracy using transfer learning (ResNet, VGG)

# Author

   Prasad Punjaram Bhosale

- Email: prasadbhosale9970@gmail.com
- GitHub: https://github.com/prasad455
- LinkedIn: www.linkedin.com/in/prasad-bhosale-99a97921a

# Acknowledgements

- Kaggle FER Dataset(https://www.kaggle.com/datasets/msambare/fer2013)
- OpenCV documentation
- TensorFlow and Keras guides

---


> If you like this project, give it a ⭐ on GitHub!
