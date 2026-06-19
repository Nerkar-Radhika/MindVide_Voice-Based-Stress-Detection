---
title: MindVibe Voice-based-Stress-Detection
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
  - streamlit
pinned: false
short_description: Voice-based psychological stress detection using ML
license: mit
---

# 🎙️ MindVibe — Voice Based Psychological Stress Detection

MindVibe detects psychological stress from voice using Machine Learning. Upload an audio file or record your voice live, and the app predicts whether you sound **Stressed** or **Calm**, along with confidence percentage and personalized stress relief suggestions.

## Features
- Supports Hindi, English, and Marathi
- Live voice recording or file upload
- Stacking Ensemble model (Random Forest + XGBoost + SVM)
- 96.23% test accuracy, 97.42% cross-validation accuracy

## Tech Stack
Python, Librosa, Scikit-learn, XGBoost, Streamlit, Docker

## Disclaimer
This tool is for informational purposes only and is not a medical diagnosis.
