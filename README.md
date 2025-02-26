# Spotify Hit Prediction

## Overview
This project applies machine learning techniques to predict hit songs on Spotify based on audio features. Using a dataset of 10,000 tracks with 15 distinct audio features, various models were trained and evaluated to determine the most accurate and efficient approach.

## Features
- Predicts whether a song will be a hit or not using machine learning models.
- Utilizes 10,000 songs with 15 extracted audio features.
- Achieves high accuracy in hit prediction.

## Machine Learning Models Used
- **Logistic Regression**: Achieved the highest accuracy of 93%.
- **Support Vector Machine (SVM)**: Demonstrated the lowest memory consumption (0.5 MiB) and efficient training times.
- Other models were explored but did not perform as well in terms of accuracy and efficiency.

## Dataset
- The dataset consists of 10,000 Spotify tracks.
- Each track is represented by 15 audio features such as tempo, energy, danceability, and more.

## Results
- **Logistic Regression** was the most accurate model at 93%.
- **SVM** was the most resource-efficient model, consuming the least memory while maintaining good performance.

## Installation and Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/spotify-hit-prediction.git
   cd spotify-hit-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model training and prediction script:
   ```sh
   python train_model.py
   ```

## Requirements
- Python 3.8+
- Pandas, NumPy, Scikit-Learn, Matplotlib, and other dependencies listed in `requirements.txt`.


