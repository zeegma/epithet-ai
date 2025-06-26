
import joblib

from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

TRAINED_MODEL = "../../models/creativity_model.keras"

def train_creativity(X, y):

    # Normalize features for the model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to use for predicction
    joblib.dump(scaler, "scaler.save")

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = None

    # Check if model exists or not
    try:
        # Load existing trained model
        model = load_model(TRAINED_MODEL)
    except ValueError:

        # Build a basic feedforward neural network
        model = Sequential([
            Dense(64, input_dim=X.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for regression
        ])

        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model with the following validation
        model.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=16)

        # Save the trained model
        model.save(TRAINED_MODEL)

    # Predict and evaluate using the testing split
    predictions = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, predictions)
    print(f"\nMean Squared Error on test set: {mse:.4f}\n")