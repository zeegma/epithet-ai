import torch
import numpy as np
import os
from core.models.personality_nn import PersonalityNN


def predict_personality(answers):
    # Personality type labels
    personality_labels = [
        "Artista",
        "Diva",
        "OA",
        "Wildcard",
        "Achiever",
        "EMO",
        "Gamer",
        "Softie",
    ]

    # Convert input to tensor of ints
    X = np.array([answers])
    X_tensor = torch.tensor(X, dtype=torch.long)

    # Load model and weights
    model = PersonalityNN()
    model_path = os.path.join("models", "personality_model_best.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict using model
    with torch.no_grad():
        logits = model(X_tensor)
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)
        output = probabilities.numpy().flatten()

    # Map output to personality label
    result = dict(zip(personality_labels, output))

    # Get the dominant personality
    max_personality = max(result.items(), key=lambda x: x[1])

    print("Personality Scores:")
    for personality, score in result.items():
        print(f"  {personality}: {score:.4f}")
    print(
        f"\nDominant personality: {max_personality[0]} (score: {max_personality[1]:.4f})"
    )

    return max_personality[0]
