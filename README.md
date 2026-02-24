# 🕵️ Fake Profile Detection System (TRUST NET)

## 📌 Overview
**TRUST NET** is a machine learning–based system designed to automatically detect fake or suspicious social media profiles. The project analyzes user activity patterns and **numerical profile attributes** to identify potentially fraudulent accounts, helping promote a safer and more trustworthy online environment.

---

## 🎯 Objectives
- Detect fake social media profiles using machine learning  
- Prioritize **numerical behavioral features** such as followers, following, posts, and account age  
- Reduce misinformation, scams, and online fraud  
- Provide an interactive and user-friendly interface using **Streamlit**

---

## 🧠 Features
- Synthetic dataset generation simulating real and fake profiles  
- Machine Learning model using **Random Forest Classifier**  
- Strong emphasis on **numeric profile attributes** over textual bio content  
- Interactive prediction interface built with **Streamlit**  
- Displays prediction probability (Fake vs Real)  
- Model performance metrics: **Accuracy** and **ROC-AUC**

---

## 🗂️ Dataset
The dataset is **synthetically generated** within the application.

### Features include:
- Followers count  
- Following count  
- Number of posts  
- Account age (in days)  
- Presence of profile picture  
- External links count  
- Bio text *(low-weight influence)*

---

## 🏗️ Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Joblib**

---

## ⚙️ Machine Learning Pipeline
1. Data generation & preprocessing  
2. Feature separation:
   - Numerical features → Scaled using `StandardScaler`
   - Text features → TF-IDF with reduced influence  
3. Model training using **Random Forest Classifier**  
4. Model evaluation using **Accuracy** and **ROC-AUC**  
5. Real-time prediction via **Streamlit UI**

---

## 🚀 How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv env
