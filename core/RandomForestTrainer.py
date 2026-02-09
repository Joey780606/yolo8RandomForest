import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
import config


class RandomForestTrainer:
    """Train, evaluate, save, and load a Random Forest classifier for face recognition."""

    def __init__(self):
        self.model = None
        self.classes = None

    def train(self, X, y):
        """
        Train a Random Forest model and compute evaluation metrics.

        Args:
            X: Feature matrix of shape (n_samples, 25).
            y: Label array of shape (n_samples,).

        Returns:
            Dict with keys:
                'trainingAccuracy': float
                'cvMean': float (cross-validation mean accuracy)
                'cvStd': float (cross-validation std)
                'cvScores': array of per-fold scores
                'confusionMatrix': 2D numpy array
                'perPersonAccuracy': dict mapping name to accuracy
                'classes': array of unique class names
        """
        try:
            self.model = RandomForestClassifier(
                n_estimators=config.RfEstimators,
                random_state=config.RfRandomState,
                n_jobs=-1,
            )

            # Train on full dataset
            self.model.fit(X, y)
            self.classes = self.model.classes_

            # Training accuracy
            trainPredictions = self.model.predict(X)
            trainingAccuracy = accuracy_score(y, trainPredictions)

            # Cross-validation (handle small datasets)
            uniqueClasses, classCounts = np.unique(y, return_counts=True)
            minSamples = classCounts.min()
            nFolds = min(config.RfCvFolds, minSamples, len(y))
            nFolds = max(nFolds, 2)

            cvScores = cross_val_score(
                RandomForestClassifier(
                    n_estimators=config.RfEstimators,
                    random_state=config.RfRandomState,
                    n_jobs=-1,
                ),
                X, y, cv=nFolds, scoring="accuracy",
            )

            # Cross-validation predictions for confusion matrix
            cvPredictions = cross_val_predict(
                RandomForestClassifier(
                    n_estimators=config.RfEstimators,
                    random_state=config.RfRandomState,
                    n_jobs=-1,
                ),
                X, y, cv=nFolds,
            )

            cm = confusion_matrix(y, cvPredictions, labels=uniqueClasses)

            # Per-person accuracy from cross-validation predictions
            perPersonAccuracy = {}
            for cls in uniqueClasses:
                mask = y == cls
                if mask.sum() > 0:
                    clsAcc = accuracy_score(y[mask], cvPredictions[mask])
                    perPersonAccuracy[cls] = clsAcc

            return {
                "trainingAccuracy": trainingAccuracy,
                "cvMean": cvScores.mean(),
                "cvStd": cvScores.std(),
                "cvScores": cvScores,
                "confusionMatrix": cm,
                "perPersonAccuracy": perPersonAccuracy,
                "classes": uniqueClasses,
            }

        except Exception as e:
            print(f"[RandomForestTrainer] Error during training: {e}")
            raise

    def saveModel(self, path=None):
        """Save the trained model to disk."""
        savePath = path or config.RfModelPath
        try:
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            joblib.dump(self.model, savePath)
            print(f"[RandomForestTrainer] Model saved to {savePath}")
        except Exception as e:
            print(f"[RandomForestTrainer] Error saving model: {e}")
            raise

    def loadModel(self, path=None):
        """Load a trained model from disk."""
        loadPath = path or config.RfModelPath
        try:
            self.model = joblib.load(loadPath)
            self.classes = self.model.classes_
            print(f"[RandomForestTrainer] Model loaded from {loadPath}")
        except Exception as e:
            print(f"[RandomForestTrainer] Error loading model: {e}")
            raise

    def predict(self, features):
        """
        Predict the person name from features.

        Args:
            features: List or array of 25 feature values.

        Returns:
            Tuple (predictedName, confidence) where confidence is the max class probability.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or loadModel() first.")

        try:
            X = np.array(features).reshape(1, -1)
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities.max()
            return prediction, confidence
        except Exception as e:
            print(f"[RandomForestTrainer] Error during prediction: {e}")
            raise

    def isModelLoaded(self):
        """Check if a model is currently loaded."""
        return self.model is not None
