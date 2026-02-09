import os
import sys

# Ensure project root is on path so config can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.YoloFaceDetector import YoloFaceDetector
from core.RandomForestTrainer import RandomForestTrainer
from core.CameraCapture import CameraCapture
from ui.App import App


def main():
    # Create necessary directories
    os.makedirs(config.DataDir, exist_ok=True)
    os.makedirs(config.ModelsDir, exist_ok=True)

    # Initialize core components
    yoloDetector = YoloFaceDetector()
    rfTrainer = RandomForestTrainer()
    cameraCapture = CameraCapture()

    # Load RF model if it exists
    if os.path.exists(config.RfModelPath):
        try:
            rfTrainer.loadModel()
        except Exception as e:
            print(f"[main] Could not load saved RF model: {e}")

    # Create and run the application
    app = App(yoloDetector, rfTrainer, cameraCapture)
    app.protocol("WM_DELETE_WINDOW", app.onClosing)
    app.mainloop()


if __name__ == "__main__":
    main()
