import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd
import joblib
import types
import sys
import os
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

# 👇 Add AgeBinner definition so unpickling works
class AgeBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import pandas as pd
        bins = [0, 18, 30, 45, 60, 80, 120]
        labels = ['<18','18-30','30-45','45-60','60-80','80+']
        return pd.DataFrame(pd.cut(X.iloc[:, 0], bins=bins, labels=labels))
    

module_name = "src.train_model"
if module_name not in sys.modules:
    fake_module = types.ModuleType(module_name)
    fake_module.AgeBinner = AgeBinner
    sys.modules[module_name] = fake_module

# Load model (path relative to project root)
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
model_dict = joblib.load(model_path)
model = model_dict['model']
threshold = model_dict['threshold']

image_path = os.path.join(os.path.dirname(__file__), "..", "assets", "liver_icon.png")
liver_img = Image.open(image_path)
st.image(liver_img, width=150)

st.markdown("""
# 🏥 Liver Disease Prediction App
Predict whether a patient has liver disease based on lab results.

**Instructions:**  
- Fill in all fields below.  
- Click **Predict** to see the result and probability.
""")

# Input fields
age = st.number_input("Age", 1, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.number_input("Total Bilirubin")
direct_bilirubin = st.number_input("Direct Bilirubin")
alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
total_protiens = st.number_input("Total Protiens")
albumin = st.number_input("Albumin")
albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", value=1.0)

# Prediction button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alkaline_phosphotase,
        "Alamine_Aminotransferase": alamine_aminotransferase,
        "Aspartate_Aminotransferase": aspartate_aminotransferase,
        "Total_Protiens": total_protiens,
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": albumin_globulin_ratio
    }])

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = 1 if prob >= threshold else 0
    st.write("Predicted Class:", "Liver Disease" if pred == 1 else "No Disease")
    st.write("Probability:", prob)
st.markdown("*Note: Model predictions are for educational purposes.*")