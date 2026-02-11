import os
import cv2
import numpy as np
import numpy.typing as npt
import logging
from typing import Iterator, Sequence, Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split

from src.classifier.classifier import train_model, evaluate_model, save_model
from src.tracking.hand_tracking import HandDetector

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

VIDEO_EXTS = (".mov", ".mp4", ".mkv", ".avi")

LM_DIR = os.path.join(os.getcwd(), 'data', 'landmarks')
os.makedirs(LM_DIR, exist_ok=True)

def _iter_video_files(label_path: str) -> Iterator[str]:
    """Iterate over valid video files in a directory.

    Args:
        label_path (str): Path to the directory containing video files.

    Yields:
        str: Full path to each valid video file.
    """
    for fname in os.listdir(label_path):
        if fname.startswith("."):
            continue
        if fname.lower().endswith(VIDEO_EXTS):
            yield os.path.join(label_path, fname)

def _hand_to_vector(
        hand_lm_list: Sequence[Tuple[int, float, float]]
    ) -> Optional[npt.NDArray]:
    """Convert hand landmarks into a normalized feature vector.

    Landmarks are translation and scale invariant by anchoring at the
    wrist (id 0) and scaling by the wrist → middle-finger MCP distance (id 9).

    Args:
        hand_lm_list (Sequence[Tuple[int, float, float]]):
            Iterable of (landmark_id, x, y) coordinates.

    Returns:
        npt.NDArray[np.float32] | None:
            Flattened normalized vector [x1, y1, x2, y2, ...].
            Returns None if normalization scale is zero.
    """
    hand_lm_list = sorted(hand_lm_list, key=lambda row: row[0])

    wx, wy = hand_lm_list[0][1], hand_lm_list[0][2]
    mx, my = hand_lm_list[9][1], hand_lm_list[9][2]
    scale = ((mx - wx) ** 2 + (my - wy) ** 2) ** 0.5
    if scale == 0:
        return None

    vec = []
    for _id, cx, cy in hand_lm_list:
        vec.extend([(cx - wx) / scale, (cy - wy) / scale])

    return np.array(vec, dtype=np.float32)


def _extract_landmarks_from_video_flat(
    detector: HandDetector,
    video_path: str,
    out_label_dir: str
) -> int:
    """Extract normalized hand landmark vectors from a video.

    Processes each frame, detects hands, converts each detected hand
    into a normalized feature vector, and saves it as a .npy file.

    Args:
        detector (HandDetector): Hand detection / landmark extraction model.
        video_path (str): Path to the input video file.
        out_label_dir (str): Directory where frame vectors will be saved.

    Returns:
        int: Number of successfully saved frame vectors.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.info(f"[WARN] Could not open video: {video_path}")
        return 0

    os.makedirs(out_label_dir, exist_ok=True)

    video_base = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detector.find_hands(frame, draw=False)
        lm_list = detector.find_pos(frame, draw=False)

        if lm_list:
            for hand_idx, hand_landmarks in enumerate(lm_list):

                if len(hand_landmarks) != 21:
                    continue

                vec = _hand_to_vector(hand_landmarks)
                if vec is None:
                    continue

                out_path = os.path.join(
                    out_label_dir,
                    f"{video_base}_frame_{frame_idx:06d}_hand_{hand_idx}.npy"
                )

                np.save(out_path, vec)
                saved += 1

        frame_idx += 1

    cap.release()
    return saved

def extract_landmarks(video_dir: str):
    """Extract landmark positions from input video files

    Args:
        video_dir (str): Path to data directory
    """

    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    total_saved = 0

    for label in os.listdir(video_dir):
        if label.startswith("."):
            continue

        label_path = os.path.join(video_dir, label)
        if not os.path.isdir(label_path):
            continue

        out_label_dir = os.path.join(LM_DIR, label)
        os.makedirs(out_label_dir, exist_ok=True)

        logger.info(f"\n=== Label: {label} ===")

        for video_path in _iter_video_files(label_path):
            saved = _extract_landmarks_from_video_flat(
                detector=detector,
                video_path=video_path,
                out_label_dir=out_label_dir
            )
            logger.info(f"  {os.path.basename(video_path)} -> saved {saved} frames")
            total_saved += saved

    logger.info(f"\nDONE. Total landmark vectors saved: {total_saved}")

def split_data() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Splits the data into X and y, then uses sklearn.model_selection.train_test_split to split into train and test data.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]: X_train, X_test, y_train, y_test
    """
    per_label = defaultdict(list)

    for label in os.listdir(LM_DIR):
        if label.startswith("."):
            continue
        label_path = os.path.join(LM_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for fname in os.listdir(label_path):
            if fname.startswith(".") or not fname.endswith(".npy"):
                continue
            per_label[label].append(os.path.join(label_path, fname))

    counts = {k: len(v) for k, v in per_label.items()}
    logger.info("Counts before:", counts)

    min_count = min(counts.values())

    for label in per_label:
        np.random.shuffle(per_label[label])
        per_label[label] = per_label[label][:min_count]

    X = []
    y = []

    for label, files in per_label.items():
        for path in files:
            X.append(np.load(path))
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    logger.info("Counts after:", {k: min_count for k in per_label})
    logger.info("X shape:", X.shape)


    logger.info("Loaded:", X.shape, y.shape)  # (num_samples, 42) (num_samples,)

    # ---- train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    return X_train, X_test, y_train, y_test

def run_pipeline(
        video_dir: str,
        skip_eval: bool = True,
    ) -> None:
    """Runs full data processing + classification model training pipeline

    Args:
        video_dir (str): Path to training data directory (labeled video files)
        skip_eval (bool, optional): Skip evaluation metrics. Defaults to True.
    """
    if video_dir is None:
        video_dir = os.path.join(os.getcwd(), 'data', 'video')

    if not os.path.isdir(video_dir):
        logger.error("Missing data directory")
        return
    
    extract_landmarks(video_dir)
    X_train, X_test, y_train, y_test = split_data()
    model = train_model(X_train, y_train)
    if not skip_eval:
        evaluate_model(model, X_test, y_test)
    save_model(model)
    