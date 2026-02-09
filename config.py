import os

# Base directory (project root)
BaseDir = os.path.dirname(os.path.abspath(__file__))

# Directory paths
DataDir = os.path.join(BaseDir, "data")
ModelsDir = os.path.join(BaseDir, "models")

# File paths
DatasetPath = os.path.join(DataDir, "WhoDataset.csv")
RfModelPath = os.path.join(ModelsDir, "RandomForestModel.joblib")
YoloModelPath = os.path.join(ModelsDir, "yolov8n-face.pt")

# YOLO settings
YoloConfidence = 0.5
YoloImageSize = 640

# Random Forest settings
RfEstimators = 100
RfCvFolds = 5
RfRandomState = 42

# Dynamic recognition settings
RecognitionTimeoutSeconds = 10
RecognitionMaxDetections = 30
RecognitionMatchThreshold = 20
RecognitionDetectionIntervalMs = 333

# Camera settings
CameraIndex = 0
CameraFps = 30

# Supported image extensions
ImageExtensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Feature column names (25 features)
DistanceFeatureNames = [
    "DistLeftEyeRightEye",
    "DistLeftEyeNose",
    "DistLeftEyeLeftMouth",
    "DistLeftEyeRightMouth",
    "DistRightEyeNose",
    "DistRightEyeLeftMouth",
    "DistRightEyeRightMouth",
    "DistNoseLeftMouth",
    "DistNoseRightMouth",
    "DistLeftMouthRightMouth",
]

PositionFeatureNames = [
    "RelPosLeftEyeX", "RelPosLeftEyeY",
    "RelPosRightEyeX", "RelPosRightEyeY",
    "RelPosNoseX", "RelPosNoseY",
    "RelPosLeftMouthX", "RelPosLeftMouthY",
    "RelPosRightMouthX", "RelPosRightMouthY",
]

RatioFeatureNames = [
    "RatioInterEyeToNoseMouth",
    "RatioEyeWidthToMouthWidth",
    "RatioNoseAsymmetry",
    "RatioUpperToLowerFace",
    "RatioLeftToRightSide",
]

AllFeatureNames = DistanceFeatureNames + PositionFeatureNames + RatioFeatureNames
