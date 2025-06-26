
import pandas as pd
import numpy as np

from train import train_creativity
from numpy.linalg import norm

MODEL_PATH = "creativity_embedded_dataset.npz"
WORD_VECTOR_PATH = "word_vectors.npz"
USERNAME_DATA = "username_data.csv"


def main():
    
    # Set X and y initially to None
    X, y = None, None

    # If there is an existing embedded dataset, load it
    try:
        data = np.load(MODEL_PATH)
        X, y = data["X"], data["Y"]

    # If no embedded dataset exists, create one from training data
    except FileNotFoundError:
        training_data = parser(USERNAME_DATA)
        X, y = get_embedded_data(training_data)

    # Train the model
    train_creativity(X, y)


# Prepare all training data for training the model
def get_embedded_data(data):

    # Store all the usernames from the training data
    usernames = []

    # Input features
    embeddings = []
    cosine_scores = []
    alliteration_scores = []
    length = []

    # Output label
    label = []

    # Iterate through the dataframe
    for index, row in data.iterrows():

        # Get the word values
        word1 = row.iat[0].lower().strip()
        word2 = row.iat[1].lower().strip()

        # Append the words to act as username
        usernames.append([word1, word2])

        # Get the creativity label of the username
        label.append(row.iat[2])

        # Check if the two words have same starting letter or not
        alliteration_scores.append(1 if word1[0] == word2[0] else 0)

        # Append length of the concatenated words
        length.append(len(word1+word2))

    # Load stored word vectors
    word_vectors = np.load(WORD_VECTOR_PATH)

    # Loop through the usernames and get their 
    # vector embeddings and cosine similarity
    for username in usernames:
        vec = embed_username(username, word_vectors)

        # Compute for the mean of two vectors
        mean_embedding = np.mean([vec[0], vec[1]], axis=0)
        embeddings.append(mean_embedding)

        # Compute the cosine similarity of two vectors
        cs = cosine_similarity(vec[0], vec[1])
        cosine_scores.append(cs)

    # Combine data into a feature variable
    X = []
    for i in range(len(usernames)):
        features = [cosine_scores[i]]
        features.extend(embeddings[i])
        X.append(features)

    # Set X and Y as input and output variables
    X = np.array(X)
    Y = np.array(label)

    print("Created embedded dataset.")

    # Save preprocessed data
    np.savez(MODEL_PATH, X=X, Y=Y)

    return X, Y


# Parse the main training csv and return a dataframe
def parser(filename):
    dataframe = pd.read_csv(filename, index_col=0) 

    return dataframe


# Compute the words similarity in the 50D vector
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# Gets the username word's embeddings
def embed_username(words, word_vectors):
    vectors = []
    for word in words:
        vec = word_vectors[word]

        if vec is not None:
            vectors.append(vec)

    return vectors

if __name__ == "__main__":
    main()

