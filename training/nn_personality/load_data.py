import pandas as pd
import os


def load_data():
    # Load dataset
    path = os.path.join(
        os.path.dirname(__file__), "../../data/personality_quiz_dataset.csv"
    )
    df = pd.read_csv(path)

    # Inputs: Q1 to Q15 answers
    X = df[[f"Q{i}" for i in range(1, 16)]].values

    # Targets: 8 softmax outputs per row
    y = df[
        ["Artista", "Diva", "OA", "Wildcard", "Achiever", "EMO", "Gamer", "Softie"]
    ].values

    return X, y
