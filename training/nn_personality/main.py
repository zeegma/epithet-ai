import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from training.nn_personality.train import train
from core.predict.personality import predict_personality


def do_train():
    train()


def do_predict():
    # Sample quiz input
    # Acelle
    # answers = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 4]

    # Ti met
    # answers = [1, 1, 4, 4, 4, 1, 4, 1, 2, 2, 4, 4, 1, 4, 4]

    # Syruz Ken
    # answers = [3, 1, 3, 4, 2, 4, 4, 2, 3, 2, 3, 4, 4, 1, 3]

    # Chrysler
    # answers = [1, 2, 3, 4, 2, 2, 4, 3, 1, 4, 1, 3, 4, 4, 1]

    # Fervicmar
    # answers = [2, 4, 2, 3, 2, 3, 3, 2, 4, 1, 2, 2, 4, 3, 2]

    # Henry
    # answers = [1, 4, 4, 2. 3, 1, 2, 4, 3, 2, 2, 1, 2, 3, 3, 3]

    # Princess Jane
    # answers = [3, 1, 1, 4, 1, 1, 3, 2, 3, 1, 1, 2, 1, 1, 3]

    # Hands
    answers = [3, 1, 1, 1, 1, 1, 3, 2, 1, 2, 1, 3, 4, 1, 1]

    # Predict using imported predict function
    _ = predict_personality(answers)


def main():
    if len(sys.argv) < 2:
        print("> Enter: python main.py [train|predict]")
        return

    command = sys.argv[1]

    if command == "train":
        do_train()
    elif command == "predict":
        do_predict()
    else:
        print(f"> Unknown command: {command}")


if __name__ == "__main__":
    main()
