from sklearn.preprocessing import MinMaxScaler


def preprocess(X, scaler=None):
    if scaler is None:
        # Training: fit new scaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        # Prediction: use existing scaler
        return scaler.transform(X)
