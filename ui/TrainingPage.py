import threading
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox

from core import DatasetManager


class TrainingPage(ctk.CTkFrame):
    """Page 2: Random Forest training with metrics display."""

    def __init__(self, parent, rfTrainer):
        super().__init__(parent)
        self.rfTrainer = rfTrainer
        self.isTraining = False

        self._buildUi()

    def _buildUi(self):
        """Build the UI layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Top section: Dataset info and train button
        topFrame = ctk.CTkFrame(self)
        topFrame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        topFrame.grid_columnconfigure(0, weight=1)

        headerLabel = ctk.CTkLabel(
            topFrame, text="Random Forest Training",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        headerLabel.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Dataset sample counts display
        self.sampleInfoLabel = ctk.CTkLabel(
            topFrame, text="Dataset: Not loaded",
            font=ctk.CTkFont(size=13), justify="left", anchor="w",
        )
        self.sampleInfoLabel.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

        # Train button
        buttonFrame = ctk.CTkFrame(topFrame, fg_color="transparent")
        buttonFrame.grid(row=2, column=0, padx=15, pady=(5, 15), sticky="ew")

        self.trainBtn = ctk.CTkButton(
            buttonFrame, text="Train Random Forest",
            command=self._onTrain,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45, width=200,
        )
        self.trainBtn.pack(side="left")

        self.statusLabel = ctk.CTkLabel(
            buttonFrame, text="",
            font=ctk.CTkFont(size=13),
        )
        self.statusLabel.pack(side="left", padx=20)

        # Results section (scrollable)
        resultsFrame = ctk.CTkScrollableFrame(self)
        resultsFrame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        resultsFrame.grid_columnconfigure(0, weight=1)

        # Accuracy metrics
        self.accuracyLabel = ctk.CTkLabel(
            resultsFrame, text="",
            font=ctk.CTkFont(size=14), justify="left", anchor="w",
        )
        self.accuracyLabel.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Per-person accuracy table
        self.perPersonLabel = ctk.CTkLabel(
            resultsFrame, text="",
            font=ctk.CTkFont(size=13, family="Courier"),
            justify="left", anchor="w",
        )
        self.perPersonLabel.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Confusion matrix image
        self.cmImageLabel = ctk.CTkLabel(resultsFrame, text="")
        self.cmImageLabel.grid(row=2, column=0, padx=10, pady=10)

    def onShow(self):
        """Called when page is shown; refresh dataset info."""
        self._refreshSampleInfo()

    def _refreshSampleInfo(self):
        """Update the sample count info display."""
        counts = DatasetManager.getSampleCountsPerPerson()
        if not counts:
            self.sampleInfoLabel.configure(text="Dataset: Empty (no samples found)")
            return

        totalSamples = sum(counts.values())
        lines = [f"Dataset: {totalSamples} total samples, {len(counts)} person(s)"]
        for personName, count in sorted(counts.items()):
            lines.append(f"  - {personName}: {count} samples")
        self.sampleInfoLabel.configure(text="\n".join(lines))

    def _onTrain(self):
        """Handle Train button press."""
        if self.isTraining:
            messagebox.showinfo("In Progress", "Training is already running.")
            return

        X, y = DatasetManager.getTrainingData()
        if X is None or len(X) == 0:
            messagebox.showwarning(
                "No Data",
                "No training data found. Please extract features first.",
            )
            return

        uniqueClasses = np.unique(y)
        if len(uniqueClasses) < 2:
            messagebox.showwarning(
                "Insufficient Data",
                "Need at least 2 different persons in the dataset for training.",
            )
            return

        self.isTraining = True
        self.trainBtn.configure(state="disabled")
        self.statusLabel.configure(text="Training in progress...")

        thread = threading.Thread(
            target=self._runTraining, args=(X, y), daemon=True,
        )
        thread.start()

    def _runTraining(self, X, y):
        """Run RF training in background thread."""
        try:
            metrics = self.rfTrainer.train(X, y)
            self.rfTrainer.saveModel()

            # Generate confusion matrix plot
            cmImage = self._generateConfusionMatrixImage(
                metrics["confusionMatrix"], metrics["classes"],
            )

            self.after(0, lambda: self._displayResults(metrics, cmImage))

        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror(
                "Training Error", f"Training failed: {err}",
            ))
        finally:
            self.after(0, self._resetUi)

    def _displayResults(self, metrics, cmImage):
        """Display training results in the UI."""
        # Accuracy metrics
        accuracyText = (
            f"Training Accuracy: {metrics['trainingAccuracy']:.4f}\n"
            f"Cross-Validation Accuracy: {metrics['cvMean']:.4f} "
            f"(+/- {metrics['cvStd']:.4f})\n"
            f"CV Fold Scores: {', '.join(f'{s:.3f}' for s in metrics['cvScores'])}"
        )
        self.accuracyLabel.configure(text=accuracyText)

        # Per-person accuracy table
        perPerson = metrics["perPersonAccuracy"]
        if perPerson:
            lines = ["Per-Person Accuracy (Cross-Validation):", "-" * 40]
            maxNameLen = max(len(str(n)) for n in perPerson.keys())
            for personName, acc in sorted(perPerson.items()):
                lines.append(f"  {str(personName):<{maxNameLen}}  {acc:.4f}")
            self.perPersonLabel.configure(text="\n".join(lines))

        # Confusion matrix image
        if cmImage is not None:
            ctkImage = ctk.CTkImage(light_image=cmImage, size=cmImage.size)
            self.cmImageLabel.configure(image=ctkImage, text="")
            self.cmImageLabel._ctkImage = ctkImage

        self.statusLabel.configure(text="Training complete! Model saved.")

        messagebox.showinfo(
            "Training Complete",
            f"Random Forest training finished.\n"
            f"Training Accuracy: {metrics['trainingAccuracy']:.4f}\n"
            f"CV Accuracy: {metrics['cvMean']:.4f} (+/- {metrics['cvStd']:.4f})",
        )

    def _generateConfusionMatrixImage(self, cm, classes):
        """Generate a confusion matrix heatmap as a PIL Image."""
        try:
            fig, ax = plt.subplots(figsize=(max(4, len(classes) * 0.8 + 2),
                                            max(3, len(classes) * 0.6 + 2)))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=classes,
                yticklabels=classes,
                xlabel="Predicted",
                ylabel="Actual",
                title="Confusion Matrix (Cross-Validation)",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Add text annotations
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )

            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).copy()

        except Exception as e:
            print(f"[TrainingPage] Error generating confusion matrix: {e}")
            return None

    def _resetUi(self):
        """Reset UI after training."""
        self.isTraining = False
        self.trainBtn.configure(state="normal")
        self._refreshSampleInfo()
