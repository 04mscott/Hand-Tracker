from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import numpy.typing as npt
import logging
import os

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

MODEL_PATH = os.path.join(os.getcwd(), 'classifier', 'model.pkl')

def save_model(model: DecisionTreeClassifier) -> None:
    """Saves model to /classifier/model.pkl

    Args:
        model (DecisionTreeClassifier): Fitted decision tree classifier to save
    """
    joblib.dump({"model": model, "labels": model.classes_}, MODEL_PATH)
    logger.info(f"Model Successfully Saved to {MODEL_PATH}")

def train_model(
        X_train: npt.NDArray,
        y_train: npt.NDArray,
    ) -> DecisionTreeClassifier:
    """Train the classification model that identifies the hand poses during live hand recognition

    Args:
        X_train (npt.NDArray): Training data
        y_train (npt.NDArray): Training labels
    """
    try:
        logger.info("Training classification model...")
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=8,
            min_samples_leaf=10
        )
        model.fit(X_train, y_train)
        logger.info("Successfully trained classificaiton model")
        return model
    except Exception as e:
        logger.error(f"Error training classification model, {e}")

def evaluate_model(
        model: DecisionTreeClassifier,
        X_test: npt.NDArray,
        y_test: npt.NDArray,
    ) -> None:
    """Evaluate and save classification model

    Args:
        model (DecisionTreeClassifier): Fitted decision tree classifier
        X_test (npt.NDArray): Test data
        y_test (npt.NDArray): Test labels
        log_report (bool, optional): Detailed logging evaluation report. Defaults to False.

    Returns:
        tuple[npt.NDArray, str, npt.NDArray]: Confusion matrix, evaluation report, unique labels
    """
    labels = getattr(model, "classes_", None)
    if labels is None:
        labels = np.unique(y_test)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_str = classification_report(y_test, y_pred)

    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report_str)

    try:
        save_model(model)
    except Exception as e:
        logger.error(f"Error saving classifiction model, {e}")

    return cm, report_str, labels