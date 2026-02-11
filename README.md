# Hand Gesture Recognition via Landmark-Based Feature Learning

## Overview

This project implements a hand gesture recognition system using landmark-based feature extraction and supervised classification. Hand landmarks are detected from video frames using OpenCV and MediaPipe, normalized into invariant feature vectors, and used to train a machine learning classifier.

## Data Pipeline

The system follows a structured preprocessing workflow:

1. Input videos are organized by gesture label.
2. Videos are processed frame by frame.
3. Hand landmarks are extracted using MediaPipe.
4. Landmarks are converted into translation and scale invariant feature vectors.
5. Feature vectors are stored as NumPy arrays.
6. Samples are balanced across labels.
7. Data is split into training and test sets.

This design allows deterministic dataset generation and efficient retraining as new data is added.

## Model

A supervised classification model is trained directly on the normalized landmark vectors. The model predicts gestures based solely on hand geometry rather than raw pixel data, reducing sensitivity to lighting, background variation, and camera differences.

## Future Work

Planned extensions include enabling gesture and motion based controls through computer vision hand tracking. The initial demonstration target is a basic 3D renderer, with the broader goal of supporting interaction across multiple applications.
