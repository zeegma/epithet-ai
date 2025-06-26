import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from core.models.personality_nn import PersonalityNN
from training.nn_personality.load_data import load_data
from training.nn_personality.preprocess import preprocess


def print_section(title, width=60):
    print("\n" + "-" * width)
    print(title.center(width))
    print("-" * width)


def evaluate_model(model, loss_fn, X, y, set_name=""):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y).item()
        preds = torch.argmax(y_pred, dim=1).numpy()
        true = y.numpy()

    print_section(f"{set_name.upper()} EVALUATION")
    print(f"Loss: {loss:.4f}")
    accuracy = (preds == true).mean()
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true, preds, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true, preds))

    return loss


def train():
    # Load and preprocess data
    X, y = load_data()
    X_scaled, scaler = preprocess(X)

    # Split into train (60%), temp (40%) i.e. val + test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )

    # Then split 40% temp into 50% val & 50% test (20% each in actuality)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Initialize model
    model = PersonalityNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Define target
    target_epochs = 50000

    # Early stopping variables
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    # Main training loop
    for epoch in range(target_epochs):
        model.train()

        # Step 1: Forward pass
        y_pred = model(X_train)

        # Step 2: Compute loss (cross entropy loss)
        loss = loss_fn(y_pred, y_train)

        # Step 3: Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Step 4: Update parameters
        optimizer.step()

        # Then just print validation tracking every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val)
            model.train()

            print(
                f"Epoch {epoch+1} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
            )

            # Early stopping logic
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save the best model and scaler
                torch.save(model.state_dict(), "models/personality_model_best.pt")
                joblib.dump(scaler, "models/personality_scaler_best.pkl")
                print(f"> New best model saved >>> Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} - Best val loss: {best_val_loss:.4f}"
                )
                break

    # Save final model and scaler (in case early stopping didn't trigger)
    torch.save(model.state_dict(), "models/personality_model_final.pt")
    joblib.dump(scaler, "models/personality_scaler_final.pkl")

    # Final evaluation flow
    print_section("TRAINING COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Training epochs: {epoch + 1}")

    # Evaluate all sets
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY".center(60))
    print("=" * 60)

    _ = evaluate_model(model, loss_fn, X_val, y_val, "Validation")
    _ = evaluate_model(model, loss_fn, X_test, y_test, "Test")
