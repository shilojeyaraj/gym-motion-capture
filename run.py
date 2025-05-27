try:
    import mediapipe as mp
    import cv2
    import tensorflow as tf
    print("TensorFlow is installed. Version:", tf.__version__)
except ImportError:
    print("TensorFlow is NOT installed.")