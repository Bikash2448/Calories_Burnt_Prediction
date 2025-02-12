# 🚀 Machine Learning Model Deployment with Flask  

This project is a **Flask-based web application** that serves a **Random Forest Regression model** for predicting **calories burned** based on user exercise data. The model was trained using **Scikit-Learn**, and the API is deployed using **Render**.  

---

## 📌 Features  
- 🏋️ **Predicts Calories Burned** based on input features.  
- 🖥️ **Flask API** with a user-friendly web interface.  
- 📦 **Model Serialization** using `pickle`.  
- 🚀 **Deployed on Render** for easy access.  

---

## ⚙️ Tech Stack  
- **Python 3.12**  
- **Flask** (Backend API)  
- **Scikit-Learn** (Machine Learning Model)  
- **Pandas, NumPy, Matplotlib, Seaborn** (Data Processing & Visualization)  
- **Gunicorn & Waitress** (Production-ready deployment)  
- **Render** (Cloud Deployment)


## 📂 Project Structure  

/calories burned Prediction
│── app.py # Flask Web API
│── model.py # Machine Learning Model Training
│── random_forest_model.pkl # Serialized ML Model
│── templates/ # HTML Templates for Web Interface
│ ├── index.html # Input Form
│ ├── results.html # Prediction Output
│── requirements.txt # Required Python Packages
│── Procfile # Render Deployment Configuration
│── README.md # Project Documentation

