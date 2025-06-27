import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os


class TrainingVisualizer:
    def __init__(self, save_dir="training/nn_personality/charts"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Training metrics storage
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Best model tracking
        self.best_epoch = 0
        self.best_val_loss = float("inf")

        # Setup matplotlib style
        plt.style.use("default")

    def update_metrics(
        self,
        epoch,
        train_loss,
        val_loss,
        learning_rate=None,
        train_accuracy=None,
        val_accuracy=None,
    ):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if train_accuracy is not None:
            self.train_accuracies.append(train_accuracy)
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy)

        # Track best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def plot_loss_convergence(self, save=True, show=True):
        # Plot training and validation loss convergence
        plt.figure(figsize=(12, 6))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(
            self.epochs,
            self.train_losses,
            label="Training Loss",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            self.epochs,
            self.val_losses,
            label="Validation Loss",
            color="red",
            linewidth=2,
        )

        # Mark best model
        if self.best_epoch in self.epochs:
            best_idx = self.epochs.index(self.best_epoch)
            plt.scatter(
                self.best_epoch,
                self.val_losses[best_idx],
                color="gold",
                s=100,
                zorder=5,
                label=f"Best Model (Epoch {self.best_epoch})",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Model Loss Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # Learning rate subplot (if available)
        if self.learning_rates:
            plt.subplot(1, 2, 2)
            plt.plot(
                self.epochs,
                self.learning_rates,
                label="Learning Rate",
                color="green",
                linewidth=2,
            )
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale("log")

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_convergence_{timestamp}.png"
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight"
            )
            print(
                f"Loss convergence chart saved to: {os.path.join(self.save_dir, filename)}"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_accuracy_progression(self, save=True, show=True):
        # Plot training and validation accuracy progression (if available)
        if not self.train_accuracies and not self.val_accuracies:
            print("No accuracy data available to plot")
            return

        plt.figure(figsize=(10, 6))

        if self.train_accuracies:
            plt.plot(
                self.epochs,
                self.train_accuracies,
                label="Training Accuracy",
                color="blue",
                linewidth=2,
            )
        if self.val_accuracies:
            plt.plot(
                self.epochs,
                self.val_accuracies,
                label="Validation Accuracy",
                color="red",
                linewidth=2,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Progression")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_progression_{timestamp}.png"
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight"
            )
            print(
                f"Accuracy progression chart saved to: {os.path.join(self.save_dir, filename)}"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_training_summary(self, save=True, show=True):
        # Create a comprehensive training summary visualization
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Personality NN Training Summary", fontsize=16, fontweight="bold")

        # Manually define 3 subplots using subplot2grid
        ax_loss = plt.subplot2grid((2, 2), (0, 0))
        ax_overfit = plt.subplot2grid((2, 2), (0, 1))
        ax_stats = plt.subplot2grid((2, 2), (1, 1))

        fig.suptitle("Personality NN Training Summary", fontsize=16, fontweight="bold")

        # 1. Loss convergence
        ax_loss.plot(
            self.epochs,
            self.train_losses,
            label="Training Loss",
            color="blue",
            linewidth=2,
        )
        ax_loss.plot(
            self.epochs,
            self.val_losses,
            label="Validation Loss",
            color="red",
            linewidth=2,
        )
        if self.best_epoch in self.epochs:
            best_idx = self.epochs.index(self.best_epoch)
            ax_loss.scatter(
                self.best_epoch,
                self.val_losses[best_idx],
                color="gold",
                s=100,
                zorder=5,
                label=f"Best (Epoch {self.best_epoch})",
            )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss Convergence")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale("log")

        # 2. Loss difference (overfitting indicator)
        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
        ax_overfit.plot(self.epochs, loss_diff, color="purple", linewidth=2)
        ax_overfit.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax_overfit.set_xlabel("Epoch")
        ax_overfit.set_ylabel("Validation Loss - Training Loss")
        ax_overfit.set_title("Overfitting Indicator")
        ax_overfit.grid(True, alpha=0.3)

        # 4. Training statistics
        ax_stats.set_axis_off()
        ax_stats.patch.set_visible(False)

        stats_text = f"""Training Statistics:

Total Epochs: {max(self.epochs) if self.epochs else 0}
Best Epoch: {self.best_epoch}
Best Val Loss: {self.best_val_loss:.4f}

Final Train Loss: {self.train_losses[-1]:.4f}
Final Val Loss: {self.val_losses[-1]:.4f}

Loss Improvement: {((self.val_losses[0] - self.best_val_loss) / self.val_losses[0] * 100):.1f}%
"""
        ax_stats.text(
            0.0,
            1.0,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_summary_{timestamp}.png"
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight"
            )
            print(f"Training summary saved to: {os.path.join(self.save_dir, filename)}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_real_time(self, window_size=50):
        # Plot real-time training progress (for use during training)
        if len(self.epochs) < 2:
            return

        # Clear the figure
        plt.clf()

        # Show only recent data if window_size is specified
        if len(self.epochs) > window_size:
            recent_epochs = self.epochs[-window_size:]
            recent_train = self.train_losses[-window_size:]
            recent_val = self.val_losses[-window_size:]
        else:
            recent_epochs = self.epochs
            recent_train = self.train_losses
            recent_val = self.val_losses

        plt.plot(
            recent_epochs,
            recent_train,
            label="Training Loss",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            recent_epochs, recent_val, label="Validation Loss", color="red", linewidth=2
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Real-time Training Progress (Last {len(recent_epochs)} epochs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # Brief pause to update the plot
        plt.pause(0.01)

    def save_metrics_to_file(self):
        # Save training metrics to a CSV file for later analysis
        data = {
            "epoch": self.epochs,
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

        if self.learning_rates:
            data["learning_rate"] = self.learning_rates
        if self.train_accuracies:
            data["train_accuracy"] = self.train_accuracies
        if self.val_accuracies:
            data["val_accuracy"] = self.val_accuracies

        df = pd.DataFrame(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_metrics_{timestamp}.csv"
        filepath = os.path.join(self.save_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Training metrics saved to: {filepath}")

    def print_final_summary(self):
        # Print a final training summary to console
        print("\n" + "=" * 60)
        print("TRAINING VISUALIZATION SUMMARY".center(60))
        print("=" * 60)
        print(f"Total Epochs Trained: {max(self.epochs) if self.epochs else 0}")
        print(f"Best Model at Epoch: {self.best_epoch}")
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")

        if len(self.val_losses) > 1:
            improvement = (
                (self.val_losses[0] - self.best_val_loss) / self.val_losses[0] * 100
            )
            print(f"Total Loss Improvement: {improvement:.2f}%")

        print("=" * 60)
