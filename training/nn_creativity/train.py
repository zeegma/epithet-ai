
import joblib

from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping

TRAINED_MODEL = "../../models/creativity_model_final.keras"

def train_creativity(X, y):

    # Normalize features for the model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to use for predicction
    joblib.dump(scaler, "data/scaler.save")

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = None
    history = None

    # Check if model exists or not
    try:
        # Load existing trained model
        model = load_model(TRAINED_MODEL)
    except ValueError:

        # Build a basic feedforward neural network
        model = Sequential([
            Dense(256, input_dim=101, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])


        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'root_mean_squared_error'])

        early_stop = EarlyStopping(
            monitor='val_loss',    
            patience=300,           
            restore_best_weights=True 
        )

        # Train model with the following validation
        history = model.fit(X_train, y_train, epochs=2000, validation_split=0.1, batch_size=16, callbacks=[early_stop])

        # Save the trained model
        model.save(TRAINED_MODEL)

    # Predict and evaluate using the testing split
    predictions = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    if history:
        print("Final training MAE:", history.history['mae'][-1])
        print("Final validation MAE:", history.history['val_mae'][-1])
        print("Final training loss (MSE):", history.history['loss'][-1])
        print("Final validation loss (MSE):", history.history['val_loss'][-1])
