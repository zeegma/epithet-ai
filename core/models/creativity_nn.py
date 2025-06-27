import numpy as np
from keras.models import load_model

# Path directories
TRAINED_MODEL = "models/creativity_model256.keras"
WORD_VECTOR_PATH = "training/nn_creativity/vectors/word_vectors.npz"
SCALER = "training/nn_creativity/data/scaler.save"


# Main function that takes in the two words and the loaded model
def creativity_nn(username_list: list, model, scaler):
    embedded = get_embeddings(username_list)
    return predict_creativity(embedded, model, scaler)


# Convert raw username into embeddings
def get_embeddings(username_list: list):

    cosine_score = []
    embeddings = []

    for username in username_list:
        word1 = username[0].lower().strip()
        word2 = username[1].lower().strip()

        word_vectors = np.load(WORD_VECTOR_PATH)

        vec = [word_vectors[word1], word_vectors[word2]]
        embeddings.append(np.mean([vec[0], vec[1]], axis=0))

        # Input features
        cosine_score.append(cosine_similarity(vec[0], vec[1]))

    X = []
    for i in range(len(username_list)):
        features = [cosine_score[i]]
        features.extend(embeddings[i])
        X.append(features)

    return np.array(X)


# Predicts the username's creativity
def predict_creativity(X, model, scaler):

    x_scaled = scaler.transform(X)

    # Predict and evaluate using the testing split
    prediction = model.predict(x_scaled).flatten()

    return prediction


# Compute the words similarity in the 50D vector
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    model = load_model(TRAINED_MODEL)
    creativity_nn([["Baby", "Creamy"], ["Pixelated", "Shadow"]], model)
