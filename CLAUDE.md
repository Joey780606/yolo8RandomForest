# Project Context: YOLOv8 + Random Forest Facial Recognition Program

## Code Specification
- Programming Language: Python 3.13
- Yolo Version: YOLOv8-face via `ultralytics` (Reference: https://github.com/lindevs/yolov8-face)
- Other Tools: scikit-learn (RandomForestClassifier), pandas, numpy, matplotlib, joblib, Pillow
- UI uses customtkinter, OpenCV (cv2.VideoCapture).
- Variable Naming: CamelCase is used consistently
- Annotation Language: English
- Error Handling: All API calls must include a try-catch block.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `yolov8n-face.pt` in the `models/` directory.
3. Run: `python main.py`

## Project Structure
```
yolo8RandomForest/
├── main.py                          # Entry point: creates dirs, initializes components, launches app
├── config.py                        # Global constants: paths, YOLO/RF settings, feature column names
├── requirements.txt                 # Python dependencies
├── core/
│   ├── __init__.py
│   ├── YoloFaceDetector.py          # YOLOv8-face model wrapper (detectFaces, detectFacesFromFrame, drawLandmarksOnImage)
│   ├── FeatureEngineering.py        # 5 landmarks + bbox -> 25 normalized features (extractFeatures)
│   ├── DatasetManager.py            # WhoDataset.csv read/write (appendRecord, getSampleCountsPerPerson, getTrainingData)
│   ├── RandomForestTrainer.py       # RF train/evaluate/save/load/predict (train, saveModel, loadModel, predict)
│   └── CameraCapture.py            # OpenCV camera wrapper (start, stop, readFrame)
├── ui/
│   ├── __init__.py
│   ├── App.py                       # Main CTk window (1200x700) + left sidebar navigation + frame switching
│   ├── FeatureExtractionPage.py     # Page 1: name input, directory browser, image preview, YOLO extraction with review
│   ├── TrainingPage.py              # Page 2: sample counts, train button, accuracy/CV metrics, confusion matrix
│   ├── DynamicRecognitionPage.py    # Page 3: camera feed, 10s recognition, learn-new-face flow
│   └── ReviewDialog.py             # Modal 4-button dialog (Correct, Recapture, Discard, System Capture For All)
├── models/                          # yolov8n-face.pt + RandomForestModel.joblib
└── data/                            # WhoDataset.csv
```

## Architecture Decisions
- **Feature Engineering**: 25 features from 5 facial landmarks (leftEye, rightEye, nose, leftMouth, rightMouth):
  - 10 normalized pairwise distances (divided by bbox diagonal)
  - 10 relative positions (x,y of each point relative to bbox)
  - 5 key ratios (inter-eye/nose-mouth, eye-width/mouth-width, nose asymmetry, upper/lower face, left/right side)
- **Dataset Format**: `data/WhoDataset.csv` with columns: `Name` + 25 feature columns
- **Model Storage**: `models/RandomForestModel.joblib` via `joblib.dump`/`joblib.load`
- **Threading**: YOLO inference and RF training run in background threads to keep UI responsive
- **Camera Display Pipeline**: OpenCV BGR -> RGB -> PIL.Image -> CTkImage
- **Modal Dialogs**: CTkToplevel with `grab_set()` + `wait_window()`
- **Config Constants** (in `config.py`):
  - YOLO: confidence=0.5, imageSize=640
  - RF: n_estimators=100, cv_folds=5, random_state=42
  - Recognition: 10s timeout, 30 max detections, 20-match threshold, 333ms detection interval

## UI Design
- 1. Adopt a "left-side navigation: Grid Layout + Frame" approach.
- Refer to the program functions below. There are three pages: the first page is for feature extraction, the second page is for Random Forest training, and the third page is for dynamic recognition.

## Program Functionality
- 1. Feature Extraction (Photo Library):
- 1-1. UI Design: Includes a name input field, a directory selection (similar to a file dialog), an image display area, and a "Confirm" button. If both the name input field and the directory selection have values, pressing the "Confirm" button will automatically extract YOLO features from all images in the directory.
- 1-2. YOLO feature extraction will capture five additional points: left eye, right eye, nose tip, left corner of mouth, and right corner of mouth. After extraction, the coordinates are transformed and compared with the user's name, converting the data into a "proportion" or "relative distance" to avoid being affected by the distance of the person in the photo. Save to WhoDataset.csv.
- 1-3. During feature extraction, the default setting allows users to view the photo through a UI, and see the location of feature points automatically captured by YOLO (marked on the photo). A dialog box will pop up with four buttons: "Correct," "Recapture," "Discard this photo," and "System capture for all." If "Correct" is selected, the image is saved to WhoDataset.csv. If "Recapture" is selected, it means the feature points were captured incorrectly, and the capture will be repeated. If "Discard this photo" is selected, this photo will not be selected, and the system will automatically move to the next photo. If "System capture for all" is selected, it means that subsequent images will be saved directly to the computer without requiring user intervention.

- 2. Random Forest Training
- 2-1. The UI needs one button to execute, and some UI to show number of samples per person in the dataset, training accuracy / cross-validation score after training, and A simple confusion matrix or per-person accuracy.
- 2-2. Clicking the button will initiate Random Forest training. For information on WhoDataset.csv, see `from sklearn.ensemble import RandomForestClassifier`. After Random Forest training, save the model (e.g., joblib.dump). Define a models/ directory to do related convention.

- 3. Dynamic Recognition
- 3-1. The UI should include a "Start Recognition" button, a view of the camera recording, a text input area for the user's name, and a "Confirm" button.
- 3-2. Pressing the "Start Recognition" button activates the computer camera. Within 10 seconds, it captures 30 detection results. Random Forest determines if 20 of these results match the same person, outputting a dialog box displaying "Match Found:" followed by the name. If no match is found within 10 seconds, a dialog box displaying "Face data not found, please test again" is displayed.
- 3-3. If a person is detected within 10 seconds, but no match is found, a dialog box displaying "User not found, please enter the name and press the 'Confirm' button to continue learning" is displayed. The user enters the name of the person in the photo. While the user is entering the name, 5-10 photos from different angles can be taken quickly. After pressing the 'Confirm' button, the feature data is obtained using YOLO and stored in WhoDataset.csv along with the name. Then, Random Forest learning is performed again. After the learning is complete, a "Random Forest learning complete" dialog box will appear.
