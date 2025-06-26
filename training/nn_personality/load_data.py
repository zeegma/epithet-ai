import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def load_data():
    # Load dataset
    path = os.path.join(
        os.path.dirname(__file__), "../../data/personality_quiz_dataset.csv"
    )
    df = pd.read_csv(path)

    # Inputs: Q1 to Q15 answers
    X = df[[f"Q{i}" for i in range(1, 16)]].values

    # Encode class labels (Artista, Diva, etc.) to integers 0–7
    y = LabelEncoder().fit_transform(df["Highest Category"])

    return X, y
