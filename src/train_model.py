# src/train_model.py

import os
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from imblearn.over_sampling import SMOTE


class AgeBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        bins = [0, 18, 30, 45, 60, 80, 120]
        labels = ['<18', '18-30', '30-45', '45-60', '60-80', '80+']
        return pd.DataFrame(pd.cut(X.iloc[:,0], bins=bins, labels=labels))


age_col = ['Age']
gender_col = ['Gender']
numerical_cols = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                  'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
                  'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']


def train_model(data_path):
    """Train model and save as models/best_model.pkl"""
    
    # Load dataset
    df = pd.read_csv(data_path)

    # Features & target
    X = df.drop("Dataset", axis=1)
    y = df["Dataset"]

    # Fill NaN
    X['Albumin_and_Globulin_Ratio'] = X['Albumin_and_Globulin_Ratio'].fillna(X['Albumin_and_Globulin_Ratio'].mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Map target 1,2 → 0,1
    y_train_mapped = y_train.map({1: 0, 2: 1})
    y_test_mapped  = y_test.map({1: 0, 2: 1})

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ('boxcox', PowerTransformer(method='yeo-johnson')),  # safe for zeros/negatives
        ('scaler', StandardScaler())
    ])
    age_pipeline = Pipeline([
        ('binner', AgeBinner()),
        ('onehot', OneHotEncoder(drop='first'))
    ])
    gender_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='if_binary'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('age', age_pipeline, age_col),
        ('gender', gender_pipeline, gender_col)
    ])

    # Classifier pipeline with SMOTE
    clf_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=(y_train_mapped==0).sum()/(y_train_mapped==1).sum(),
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        ))
    ])

    # Fit
    clf_pipeline.fit(X_train, y_train_mapped)

    # Threshold search
    y_proba = clf_pipeline.predict_proba(X_test)[:,1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_recall = -1
    best_threshold = 0.5
    best_report = None
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        recall = recall_score(y_test_mapped, y_pred)
        if recall > best_recall:
            best_recall = recall
            best_threshold = thr
            best_report = classification_report(y_test_mapped, y_pred)

    print("Best threshold:", best_threshold)
    print("Best Recall:", best_recall)
    print("\nClassification Report:\n", best_report)

    # Save model
    # Ensure models folder exists
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_dict = {
        "model": clf_pipeline,
        "threshold": best_threshold
    }

    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(model_dict, model_path)
    print(f"✅ Model saved at: {model_path}")

    return clf_pipeline


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "indian_liver_patient.csv")
    train_model(data_path)
    
