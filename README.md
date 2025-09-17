# 🩺 Liver Disease Prediction App

A Machine Learning web application to predict **liver disease** based on patient health parameters.  
Built using **scikit-learn**, **Streamlit**, and deployed on **Streamlit Cloud**.  

👉 Live App: [Liver Disease Prediction](https://liver-disease-prediction-1.streamlit.app/)

---

## 🚀 Project Workflow

1. **Data Preprocessing & Training**
   - Cleaned dataset and handled missing values.
   - Applied feature engineering (e.g., Age Binning).
   - Built a scikit-learn pipeline with transformers + classifier.
   - Trained and tuned the model for best performance.

2. **Model Saving**
   - Best model was stored as a pickle (`best_model.pkl`) using `joblib`.
   - Saved both the trained pipeline and classification threshold.

3. **Streamlit App**
   - Simple and interactive UI with dropdowns, and number inputs.
   - Displays prediction (Healthy / At Risk) based on input features.
   - Custom liver icon added for visualization.

4. **Deployment**
   - Project pushed to GitHub.
   - Deployed on Streamlit Cloud for public access.

---

## 📂 Project Structure

- data
   csv file
- models/
   best_model.pkl # Saved trained model
- src/
   app.py # Streamlit app (frontend)
   train_model.py # Script to train and save model
   init.py
- requirements.txt # Dependencies
- assets
   liver_icon.png   

## 🛠️ Tech Stack
  Python 3.10+
  Pandas for data manipulation
  Scikit-learn for ML pipeline
  Joblib for saving model
  Streamlit for UI & deployment
  Pillow (PIL) for image rendering

## 📊 Learnings
Creating and saving ML pipelines with scikit-learn.
Handling custom transformers (AgeBinner) during pickle loading.
Building interactive web apps using Streamlit.
Deploying ML models to Streamlit Cloud.
