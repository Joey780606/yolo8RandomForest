import os
import time
import threading
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import customtkinter as ctk
from tkinter import messagebox

import config
from core import FeatureEngineering, DatasetManager


class DynamicRecognitionPage(ctk.CTkFrame):
    """Page 3: Dynamic face recognition using camera feed."""

    def __init__(self, parent, yoloDetector, rfTrainer, cameraCapture):
        super().__init__(parent)
        self.yoloDetector = yoloDetector
        self.rfTrainer = rfTrainer
        self.cameraCapture = cameraCapture

        self.isRecognizing = False
        self.cameraRunning = False
        self.stopEvent = threading.Event()

        self._buildUi()

    def _buildUi(self):
        """Build the UI layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top controls
        controlFrame = ctk.CTkFrame(self)
        controlFrame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        controlFrame.grid_columnconfigure(2, weight=1)

        self.startBtn = ctk.CTkButton(
            controlFrame, text="Start Recognition",
            command=self._onStartRecognition,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45, width=180,
        )
        self.startBtn.grid(row=0, column=0, padx=15, pady=15)

        self.stopBtn = ctk.CTkButton(
            controlFrame, text="Stop Camera",
            command=self.stopRecognition,
            font=ctk.CTkFont(size=14),
            height=45, width=120,
            state="disabled",
        )
        self.stopBtn.grid(row=0, column=1, padx=5, pady=15)

        self.statusLabel = ctk.CTkLabel(
            controlFrame, text="Ready. Press 'Start Recognition' to begin.",
            font=ctk.CTkFont(size=13),
        )
        self.statusLabel.grid(row=0, column=2, padx=15, pady=15, sticky="w")

        # Camera display
        self.cameraLabel = ctk.CTkLabel(
            self, text="Camera feed will appear here",
            font=ctk.CTkFont(size=14),
        )
        self.cameraLabel.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Bottom: Name input for learning
        learnFrame = ctk.CTkFrame(self)
        learnFrame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        learnFrame.grid_columnconfigure(1, weight=1)

        nameLabel = ctk.CTkLabel(
            learnFrame, text="Name:", font=ctk.CTkFont(size=14),
        )
        nameLabel.grid(row=0, column=0, padx=(15, 10), pady=15)

        self.nameEntry = ctk.CTkEntry(
            learnFrame, placeholder_text="Enter name for learning",
            font=ctk.CTkFont(size=14), height=35,
        )
        self.nameEntry.grid(row=0, column=1, padx=10, pady=15, sticky="ew")

        self.learnBtn = ctk.CTkButton(
            learnFrame, text="Confirm",
            command=self._onLearnConfirm,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40, width=100,
            state="disabled",
        )
        self.learnBtn.grid(row=0, column=2, padx=15, pady=15)

        # Internal state for learning flow
        self.capturedFrames = []
        self.awaitingNameInput = False

    def onHide(self):
        """Called when page is hidden; stop camera."""
        self.stopRecognition()

    def _onStartRecognition(self):
        """Handle Start Recognition button."""
        if not self.rfTrainer.isModelLoaded():
            # Try to load saved model
            if os.path.exists(config.RfModelPath):
                try:
                    self.rfTrainer.loadModel()
                except Exception as e:
                    messagebox.showwarning(
                        "No Model",
                        f"Could not load model: {e}\n"
                        "Please train the Random Forest first (Page 2).",
                    )
                    return
            else:
                messagebox.showwarning(
                    "No Model",
                    "No trained model found. Please train the Random Forest first (Page 2).",
                )
                return

        if self.isRecognizing:
            return

        self.isRecognizing = True
        self.stopEvent.clear()
        self.startBtn.configure(state="disabled")
        self.stopBtn.configure(state="normal")
        self.learnBtn.configure(state="disabled")
        self.awaitingNameInput = False

        thread = threading.Thread(target=self._recognitionLoop, daemon=True)
        thread.start()

    def stopRecognition(self):
        """Stop the recognition process and camera."""
        self.stopEvent.set()
        self.cameraRunning = False
        self.cameraCapture.stop()
        self.isRecognizing = False
        self.after(0, lambda: self.startBtn.configure(state="normal"))
        self.after(0, lambda: self.stopBtn.configure(state="disabled"))

    def _recognitionLoop(self):
        """Main recognition loop running in background thread."""
        # Start camera
        if not self.cameraCapture.start():
            self.after(0, lambda: messagebox.showerror(
                "Camera Error", "Could not open camera.",
            ))
            self.after(0, self._resetUi)
            return

        self.cameraRunning = True
        detectionResults = []
        faceDetectedAtLeastOnce = False
        self.capturedFrames = []

        startTime = time.time()
        lastDetectionTime = 0
        detectionIntervalSec = config.RecognitionDetectionIntervalMs / 1000.0

        self.after(0, lambda: self.statusLabel.configure(
            text="Recognizing... (10 second window)",
        ))

        while not self.stopEvent.is_set():
            elapsed = time.time() - startTime
            if elapsed > config.RecognitionTimeoutSeconds:
                break

            frame = self.cameraCapture.readFrame()
            if frame is None:
                continue

            # Display frame
            self.after(0, lambda f=frame.copy(): self._displayFrame(f))

            # Run detection at intervals
            currentTime = time.time()
            if (currentTime - lastDetectionTime >= detectionIntervalSec
                    and len(detectionResults) < config.RecognitionMaxDetections):
                lastDetectionTime = currentTime

                faces = self.yoloDetector.detectFacesFromFrame(frame)
                if faces:
                    faceDetectedAtLeastOnce = True
                    face = faces[0]
                    features = FeatureEngineering.extractFeatures(
                        face["keypoints"], face["bbox"],
                    )
                    if features is not None:
                        try:
                            prediction, confidence = self.rfTrainer.predict(features)
                            detectionResults.append(prediction)
                            self.capturedFrames.append(frame.copy())
                        except Exception:
                            pass

                remaining = config.RecognitionTimeoutSeconds - elapsed
                self.after(0, lambda r=remaining, d=len(detectionResults):
                    self.statusLabel.configure(
                        text=f"Recognizing... {r:.1f}s left, {d} detections",
                    ))

                # Check for early match
                if detectionResults:
                    counter = Counter(detectionResults)
                    mostCommon, count = counter.most_common(1)[0]
                    if count >= config.RecognitionMatchThreshold:
                        self.after(0, lambda name=mostCommon:
                            self._onMatchFound(name))
                        self._stopCameraFeed()
                        return

            # Small sleep to avoid busy loop
            time.sleep(1.0 / config.CameraFps)

        # Timeout reached - analyze results
        self._stopCameraFeed()

        if not faceDetectedAtLeastOnce:
            self.after(0, self._onNoFaceDetected)
        elif detectionResults:
            counter = Counter(detectionResults)
            mostCommon, count = counter.most_common(1)[0]
            if count >= config.RecognitionMatchThreshold:
                self.after(0, lambda name=mostCommon: self._onMatchFound(name))
            else:
                self.after(0, self._onUnrecognizedFace)
        else:
            self.after(0, self._onNoFaceDetected)

    def _stopCameraFeed(self):
        """Stop the camera after recognition loop ends."""
        self.cameraRunning = False
        self.cameraCapture.stop()
        self.after(0, lambda: self.stopBtn.configure(state="disabled"))

    def _displayFrame(self, frame):
        """Display a BGR frame in the camera label."""
        try:
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(rgbFrame)

            maxWidth = 700
            maxHeight = 450
            pilImage.thumbnail((maxWidth, maxHeight), Image.LANCZOS)

            ctkImage = ctk.CTkImage(light_image=pilImage, size=pilImage.size)
            self.cameraLabel.configure(image=ctkImage, text="")
            self.cameraLabel._ctkImage = ctkImage
        except Exception:
            pass

    def _onMatchFound(self, name):
        """Handle successful match."""
        self.isRecognizing = False
        self.startBtn.configure(state="normal")
        self.statusLabel.configure(text=f"Match found: {name}")
        messagebox.showinfo("Match Found", f"Match Found: {name}")

    def _onNoFaceDetected(self):
        """Handle no face detected within timeout."""
        self.isRecognizing = False
        self.startBtn.configure(state="normal")
        self.statusLabel.configure(text="No face detected.")
        messagebox.showinfo(
            "No Face Detected",
            "Face data not found, please test again.",
        )

    def _onUnrecognizedFace(self):
        """Handle face detected but not recognized."""
        self.isRecognizing = False
        self.awaitingNameInput = True
        self.learnBtn.configure(state="normal")
        self.startBtn.configure(state="normal")
        self.statusLabel.configure(
            text="User not found. Enter name and press Confirm to learn.",
        )
        messagebox.showinfo(
            "User Not Found",
            "User not found, please enter the name and press the "
            "'Confirm' button to continue learning.",
        )

    def _onLearnConfirm(self):
        """Handle Confirm button for learning a new face."""
        if not self.awaitingNameInput:
            return

        name = self.nameEntry.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter a person's name.")
            return

        self.learnBtn.configure(state="disabled")
        self.awaitingNameInput = False
        self.statusLabel.configure(text=f"Learning face for '{name}'...")

        thread = threading.Thread(
            target=self._runLearning, args=(name,), daemon=True,
        )
        thread.start()

    def _runLearning(self, name):
        """Run the learning process: capture frames, extract features, retrain."""
        try:
            # Use captured frames from recognition attempt + capture more
            framesToProcess = self.capturedFrames[-10:]  # Use up to 10 frames

            # Also try to capture a few more frames from camera
            if self.cameraCapture.start():
                for _ in range(5):
                    frame = self.cameraCapture.readFrame()
                    if frame is not None:
                        framesToProcess.append(frame)
                    time.sleep(0.3)
                self.cameraCapture.stop()

            if not framesToProcess:
                self.after(0, lambda: messagebox.showwarning(
                    "No Frames", "No frames available for learning.",
                ))
                return

            # Extract features from each frame
            featuresList = []
            for frame in framesToProcess:
                faces = self.yoloDetector.detectFacesFromFrame(frame)
                if faces:
                    features = FeatureEngineering.extractFeatures(
                        faces[0]["keypoints"], faces[0]["bbox"],
                    )
                    if features is not None:
                        featuresList.append(features)

            if not featuresList:
                self.after(0, lambda: messagebox.showwarning(
                    "Feature Extraction Failed",
                    "Could not extract features from captured frames.",
                ))
                return

            # Save to dataset
            DatasetManager.appendRecords(name, featuresList)

            self.after(0, lambda c=len(featuresList):
                self.statusLabel.configure(
                    text=f"Saved {c} samples for '{name}'. Retraining...",
                ))

            # Retrain
            X, y = DatasetManager.getTrainingData()
            if X is not None and len(np.unique(y)) >= 2:
                self.rfTrainer.train(X, y)
                self.rfTrainer.saveModel()
                self.after(0, lambda: messagebox.showinfo(
                    "Learning Complete",
                    "Random Forest learning complete.",
                ))
                self.after(0, lambda: self.statusLabel.configure(
                    text="Learning complete. Model retrained and saved.",
                ))
            else:
                self.after(0, lambda: messagebox.showinfo(
                    "Data Saved",
                    f"Saved {len(featuresList)} samples for '{name}'.\n"
                    "Need at least 2 persons in dataset to train. "
                    "Please add more people and train on Page 2.",
                ))

        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror(
                "Learning Error", f"Learning failed: {err}",
            ))
        finally:
            self.after(0, self._resetUi)

    def _resetUi(self):
        """Reset UI state."""
        self.isRecognizing = False
        self.startBtn.configure(state="normal")
        self.stopBtn.configure(state="disabled")
        self.learnBtn.configure(state="disabled")
