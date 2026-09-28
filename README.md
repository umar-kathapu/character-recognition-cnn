# 🔤 Character Recognition using CNN

A deep learning-based handwritten English character recognition system built using a **Convolutional Neural Network (CNN)**.

The application allows users to draw a handwritten English letter (A–Z) and predicts the character using a trained CNN model.

---

## 🚀 Live Demo

🔗 **Streamlit App:** Coming Soon

---

## 📌 Project Overview

Handwritten character recognition is a classic computer vision and deep learning problem.

This project uses a Convolutional Neural Network to recognize handwritten English alphabet characters from **A to Z**.

The trained model takes a **28 × 28 grayscale image** as input and predicts one of the 26 English alphabet classes.

A Streamlit web application provides an interactive interface where users can draw a character and receive the prediction instantly.

---

## 🎯 Objectives

- Recognize handwritten English characters from A–Z.
- Build and train a CNN-based image classification model.
- Perform image preprocessing suitable for handwritten characters.
- Evaluate the trained model using test data.
- Develop an interactive web application using Streamlit.
- Provide real-time character predictions with confidence scores.

---

## 🧠 Model Architecture

The project uses a custom Convolutional Neural Network.

```text
Input Image
    │
    ▼
28 × 28 × 1 Grayscale
    │
    ▼
Conv2D - 32 Filters + ReLU
    │
    ▼
MaxPooling
    │
    ▼
Conv2D - 64 Filters + ReLU
    │
    ▼
MaxPooling
    │
    ▼
Conv2D - 128 Filters + ReLU
    │
    ▼
Flatten
    │
    ▼
Dense - 128 + ReLU
    │
    ▼
Dropout - 0.3
    │
    ▼
Dense - 26 + Softmax
    │
    ▼
Predicted Character
