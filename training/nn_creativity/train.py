import pandas as pd
import numpy as np
import os

from numpy.linalg import norm

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

TRAINED_MODEL = "../../creativity_model.keras"

base_dir = os.path

def train_creativity(X, y):

    # 2. Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 4. Build a basic feedforward neural network
    model = Sequential([
        Dense(64, input_dim=X.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # 5. Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 6. Train model with validation
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=16)

    # 7. Save trained model
    model.save(TRAINED_MODEL)

    # 8. Predict and evaluate
    predictions = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, predictions)
    print(f"\nMean Squared Error on test set: {mse:.4f}\n")