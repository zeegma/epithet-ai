import torch
import joblib  # Add this import
from sklearn.model_selection import train_test_split
from core.models.personality_nn import PersonalityNN
from training.nn_personality.load_data import load_data
from training.nn_personality.preprocess import preprocess


def train():
    # Load and preprocess data
    X, y = load_data()
    X_scaled, scaler = preprocess(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model
    model = PersonalityNN()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define targets
    target_epochs = 20000
    target_error = 0.001  # Lowered this

    # Early stopping variables
    best_val_loss = float("inf")
    patience = 5  # Stop if no improvement for 5 checks (5000 epochs)
    patience_counter = 0

    # Main training loop
    for epoch in range(target_epochs):
        model.train()

        # Step 1: Forward pass
        y_pred = model(X_train)

        # Step 2: Compute loss (MSE)
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
                val_pred = model(X_test)
                val_loss = loss_fn(val_pred, y_test)
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

        # And early break if target loss is already met
        if loss.item() <= target_error:
            print(f"Target error reached at epoch {epoch + 1}")
            break

    # Save final model and scaler (in case early stopping didn't trigger)
    torch.save(model.state_dict(), "models/personality_model_final.pt")
    joblib.dump(scaler, "models/personality_scaler_final.pkl")

    # Final evaluation
    print("=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred, y_test)

    print(f"Final Test Loss: {test_loss.item():.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Training completed after {epoch + 1} epochs")
