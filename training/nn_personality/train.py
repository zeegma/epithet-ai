import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from core.models.personality_nn import PersonalityNN
from training.nn_personality.load_data import load_data
from training.nn_personality.chart import TrainingVisualizer


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

    # Split into train (60%), temp (40%) i.e. val + test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Then split 40% temp into 50% val & 50% test (20% each in actuality)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Convert to PyTorch tensors (as integer input for embedding layers)
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Initialize model
    model = PersonalityNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Initialize visualizer
    visualizer = TrainingVisualizer()

    # Define target
    target_epochs = 100000

    # Early stopping variables
    best_val_loss = float("inf")
    patience = 5
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

        # Then just print validation tracking every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val)

                # Calculate accuracies for visualization
                train_pred = model(X_train)
                train_acc = (
                    (torch.argmax(train_pred, dim=1) == y_train).float().mean().item()
                )
                val_acc = (torch.argmax(val_pred, dim=1) == y_val).float().mean().item()

            model.train()

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Update visualizer
            visualizer.update_metrics(
                epoch + 1, loss.item(), val_loss.item(), current_lr, train_acc, val_acc
            )

            print(
                f"Epoch {epoch + 1} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss.item())

            # Early stopping logic
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save the best model and scaler
                torch.save(model.state_dict(), "models/personality_model_best.pt")
                print(f"> New best model saved >>> Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} - Best val loss: {best_val_loss:.4f}"
                )
                break

    # Load best model before evaluation
    model.load_state_dict(torch.load("models/personality_model_best.pt"))

    # Save final model (in case early stopping didn't trigger)
    torch.save(model.state_dict(), "models/personality_model_final.pt")

    # Final evaluation flow
    print_section("TRAINING COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Training epochs: {epoch + 1}")

    # Generate all visualizations
    print("\nGenerating training visualizations...")
    visualizer.plot_training_summary()
    visualizer.plot_loss_convergence()
    visualizer.plot_accuracy_progression()
    visualizer.save_metrics_to_file()
    visualizer.print_final_summary()

    # Final evaluation (your existing code)
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY".center(60))
    print("=" * 60)

    _ = evaluate_model(model, loss_fn, X_val, y_val, "Validation")
    _ = evaluate_model(model, loss_fn, X_test, y_test, "Test")
