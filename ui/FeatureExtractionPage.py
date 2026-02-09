import os
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox

import config
from core import FeatureEngineering, DatasetManager
from ui.ReviewDialog import ReviewDialog


class FeatureExtractionPage(ctk.CTkFrame):
    """Page 1: Feature extraction from photo library using YOLO face detection."""

    def __init__(self, parent, yoloDetector):
        super().__init__(parent)
        self.yoloDetector = yoloDetector
        self.selectedDirectory = None
        self.isProcessing = False

        self._buildUi()

    def _buildUi(self):
        """Build the UI layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Top section: Name input and directory selection
        inputFrame = ctk.CTkFrame(self)
        inputFrame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        inputFrame.grid_columnconfigure(1, weight=1)

        # Name input
        nameLabel = ctk.CTkLabel(inputFrame, text="Name:", font=ctk.CTkFont(size=14))
        nameLabel.grid(row=0, column=0, padx=(15, 10), pady=10)

        self.nameEntry = ctk.CTkEntry(
            inputFrame, placeholder_text="Enter person's name",
            font=ctk.CTkFont(size=14), height=35,
        )
        self.nameEntry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Directory selection
        dirLabel = ctk.CTkLabel(inputFrame, text="Directory:", font=ctk.CTkFont(size=14))
        dirLabel.grid(row=1, column=0, padx=(15, 10), pady=10)

        dirInnerFrame = ctk.CTkFrame(inputFrame, fg_color="transparent")
        dirInnerFrame.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        dirInnerFrame.grid_columnconfigure(0, weight=1)

        self.dirEntry = ctk.CTkEntry(
            dirInnerFrame, placeholder_text="Select image directory",
            font=ctk.CTkFont(size=14), height=35, state="disabled",
        )
        self.dirEntry.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        browseBtn = ctk.CTkButton(
            dirInnerFrame, text="Browse", command=self._browseDirectory,
            width=80, height=35,
        )
        browseBtn.grid(row=0, column=1)

        # Confirm button
        self.confirmBtn = ctk.CTkButton(
            inputFrame, text="Confirm", command=self._onConfirm,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40, width=120,
        )
        self.confirmBtn.grid(row=0, column=2, rowspan=2, padx=15, pady=10)

        # Progress bar and status
        statusFrame = ctk.CTkFrame(self)
        statusFrame.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        statusFrame.grid_columnconfigure(0, weight=1)

        self.progressBar = ctk.CTkProgressBar(statusFrame)
        self.progressBar.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        self.progressBar.set(0)

        self.statusLabel = ctk.CTkLabel(
            statusFrame, text="Ready. Enter a name and select a directory to begin.",
            font=ctk.CTkFont(size=12),
        )
        self.statusLabel.grid(row=1, column=0, padx=15, pady=(0, 10))

        # Image display area
        self.imageLabel = ctk.CTkLabel(self, text="Image preview will appear here")
        self.imageLabel.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

    def _browseDirectory(self):
        """Open a directory selection dialog."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.selectedDirectory = directory
            self.dirEntry.configure(state="normal")
            self.dirEntry.delete(0, "end")
            self.dirEntry.insert(0, directory)
            self.dirEntry.configure(state="disabled")

    def _onConfirm(self):
        """Handle the Confirm button press."""
        name = self.nameEntry.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter a person's name.")
            return
        if not self.selectedDirectory:
            messagebox.showwarning("Input Required", "Please select an image directory.")
            return
        if self.isProcessing:
            messagebox.showinfo("In Progress", "Feature extraction is already running.")
            return

        self.isProcessing = True
        self.confirmBtn.configure(state="disabled")

        # Run extraction in a thread to keep UI responsive during model loading
        thread = threading.Thread(
            target=self._runExtraction, args=(name,), daemon=True,
        )
        thread.start()

    def _runExtraction(self, name):
        """Run the feature extraction process (called from background thread)."""
        try:
            # Collect image files
            imageFiles = []
            for fileName in sorted(os.listdir(self.selectedDirectory)):
                if fileName.lower().endswith(config.ImageExtensions):
                    imageFiles.append(os.path.join(self.selectedDirectory, fileName))

            if not imageFiles:
                self.after(0, lambda: messagebox.showwarning(
                    "No Images", "No image files found in the selected directory.",
                ))
                return

            totalImages = len(imageFiles)
            savedCount = 0
            autoSaveMode = False

            for idx, imagePath in enumerate(imageFiles):
                fileName = os.path.basename(imagePath)
                self.after(0, lambda i=idx, t=totalImages, f=fileName:
                    self._updateStatus(f"Processing {i + 1}/{t}: {f}"))
                self.after(0, lambda i=idx, t=totalImages:
                    self.progressBar.set((i + 1) / t))

                # Load image
                try:
                    image = cv2.imread(imagePath)
                    if image is None:
                        continue
                except Exception:
                    continue

                # Detect faces
                faces = self.yoloDetector.detectFaces(imagePath)
                if not faces:
                    self.after(0, lambda f=fileName:
                        self._updateStatus(f"No face detected in {f}, skipping."))
                    continue

                # Use the first (most confident) face
                face = faces[0]
                keypoints = face["keypoints"]
                bbox = face["bbox"]

                # Extract features
                features = FeatureEngineering.extractFeatures(keypoints, bbox)
                if features is None:
                    self.after(0, lambda f=fileName:
                        self._updateStatus(f"Feature extraction failed for {f}, skipping."))
                    continue

                # Draw landmarks on image for display
                annotatedImage = self.yoloDetector.drawLandmarksOnImage(image, [face])
                self.after(0, lambda img=annotatedImage: self._displayImage(img))

                if autoSaveMode:
                    # Auto-save without review
                    DatasetManager.appendRecord(name, features)
                    savedCount += 1
                else:
                    # Show review dialog and wait for response
                    result = self._showReviewDialogAndWait(fileName)

                    if result == "correct":
                        DatasetManager.appendRecord(name, features)
                        savedCount += 1
                    elif result == "recapture":
                        # Re-run detection on the same image
                        faces2 = self.yoloDetector.detectFaces(imagePath)
                        if faces2:
                            face2 = faces2[0]
                            features2 = FeatureEngineering.extractFeatures(
                                face2["keypoints"], face2["bbox"],
                            )
                            if features2 is not None:
                                annotated2 = self.yoloDetector.drawLandmarksOnImage(image, [face2])
                                self.after(0, lambda img=annotated2: self._displayImage(img))
                                DatasetManager.appendRecord(name, features2)
                                savedCount += 1
                    elif result == "discard":
                        pass
                    elif result == "systemCaptureAll":
                        DatasetManager.appendRecord(name, features)
                        savedCount += 1
                        autoSaveMode = True

            self.after(0, lambda s=savedCount, t=totalImages:
                self._onExtractionComplete(s, t))

        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror(
                "Error", f"Feature extraction failed: {err}",
            ))
        finally:
            self.after(0, self._resetUi)

    def _showReviewDialogAndWait(self, imageName):
        """Show the review dialog from the main thread and wait for result."""
        resultContainer = [None]
        event = threading.Event()

        def showDialog():
            dialog = ReviewDialog(self.winfo_toplevel(), imageName)
            resultContainer[0] = dialog.getResult()
            event.set()

        self.after(0, showDialog)
        event.wait()
        return resultContainer[0]

    def _displayImage(self, bgrImage):
        """Display an OpenCV BGR image in the image label."""
        try:
            rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(rgbImage)

            # Scale to fit display area
            maxWidth = 700
            maxHeight = 450
            pilImage.thumbnail((maxWidth, maxHeight), Image.LANCZOS)

            ctkImage = ctk.CTkImage(light_image=pilImage, size=pilImage.size)
            self.imageLabel.configure(image=ctkImage, text="")
            self.imageLabel._ctkImage = ctkImage  # Prevent garbage collection
        except Exception as e:
            print(f"[FeatureExtractionPage] Error displaying image: {e}")

    def _updateStatus(self, text):
        """Update the status label text."""
        self.statusLabel.configure(text=text)

    def _onExtractionComplete(self, savedCount, totalImages):
        """Handle extraction completion."""
        messagebox.showinfo(
            "Extraction Complete",
            f"Feature extraction finished.\n"
            f"Saved {savedCount} out of {totalImages} images to WhoDataset.csv.",
        )
        self._updateStatus(
            f"Complete: {savedCount}/{totalImages} images saved to dataset.",
        )

    def _resetUi(self):
        """Reset UI state after extraction."""
        self.isProcessing = False
        self.confirmBtn.configure(state="normal")
        self.progressBar.set(0)
