import cv2
import numpy as np
import config


class YoloFaceDetector:
    """Wrapper around YOLOv8-face model for face detection and landmark extraction."""

    def __init__(self, modelPath=None):
        self.modelPath = modelPath or config.YoloModelPath
        self.model = None

    def loadModel(self):
        """Load the YOLOv8-face model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.modelPath)
            print(f"[YoloFaceDetector] Model loaded from {self.modelPath}")
        except Exception as e:
            print(f"[YoloFaceDetector] Error loading model: {e}")
            raise

    def ensureModelLoaded(self):
        """Ensure the model is loaded before inference."""
        if self.model is None:
            self.loadModel()

    def detectFaces(self, imagePath):
        """
        Detect faces in an image file.

        Args:
            imagePath: Path to the image file.

        Returns:
            List of dicts, each with keys:
                'bbox': (x1, y1, x2, y2) bounding box
                'keypoints': array of shape (5, 2) with (x, y) for each landmark
                'confidence': detection confidence score
            Returns empty list if no faces detected or on error.
        """
        self.ensureModelLoaded()
        try:
            results = self.model(
                imagePath,
                conf=config.YoloConfidence,
                imgsz=config.YoloImageSize,
                verbose=False,
            )
            return self._parseResults(results)
        except Exception as e:
            print(f"[YoloFaceDetector] Error detecting faces in {imagePath}: {e}")
            return []

    def detectFacesFromFrame(self, frame):
        """
        Detect faces in an OpenCV BGR frame.

        Args:
            frame: numpy array (BGR image from OpenCV).

        Returns:
            Same format as detectFaces().
        """
        self.ensureModelLoaded()
        try:
            results = self.model(
                frame,
                conf=config.YoloConfidence,
                imgsz=config.YoloImageSize,
                verbose=False,
            )
            return self._parseResults(results)
        except Exception as e:
            print(f"[YoloFaceDetector] Error detecting faces from frame: {e}")
            return []

    def _parseResults(self, results):
        """Parse YOLO results into list of face detections."""
        faces = []
        if not results or len(results) == 0:
            return faces

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return faces

        if result.keypoints is None or result.keypoints.xy is None:
            return faces

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        keypointsAll = result.keypoints.xy.cpu().numpy()

        for i in range(len(boxes)):
            bbox = tuple(boxes[i].astype(float))
            kps = keypointsAll[i]

            # Validate keypoints: skip if all zeros (failed detection)
            if np.allclose(kps, 0):
                continue

            faces.append({
                "bbox": bbox,
                "keypoints": kps,
                "confidence": float(confidences[i]),
            })

        return faces

    def drawLandmarksOnImage(self, image, faces):
        """
        Draw bounding boxes and keypoint landmarks on an image.

        Args:
            image: numpy array (BGR image).
            faces: List of face dicts from detectFaces().

        Returns:
            Copy of image with annotations drawn.
        """
        annotated = image.copy()
        keypointColors = [
            (0, 255, 0),    # Left eye - green
            (0, 255, 0),    # Right eye - green
            (255, 0, 0),    # Nose - blue
            (0, 0, 255),    # Left mouth - red
            (0, 0, 255),    # Right mouth - red
        ]
        keypointLabels = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

            conf = face["confidence"]
            cv2.putText(
                annotated, f"{conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

            for j, kp in enumerate(face["keypoints"]):
                cx, cy = int(kp[0]), int(kp[1])
                color = keypointColors[j]
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                cv2.putText(
                    annotated, keypointLabels[j], (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        return annotated
