#!/usr/bin/env python3
"""
YOLO Model Testing Script for IDFC
Tests YOLO-based stamp and signature detection on training images
Demonstrates handling of missing model files
"""

import os
import sys
import cv2
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.preprocessing.image_loader import load_image
from src.extraction.stamp import DealerStampExtractor
from src.extraction.signature import DealerSignatureExtractor

def get_training_images(limit=5):
    """Get a sample of training images for testing"""
    train_dir = Path("data/train")
    if not train_dir.exists():
        print(f"Training directory not found: {train_dir}")
        return []

    # Get PNG files
    image_files = list(train_dir.glob("*.png"))
    return image_files[:limit]  # Limit for testing

def test_yolo_detection(image_path, detector_name, detector):
    """Test YOLO detection on a single image"""
    print(f"\n--- Testing {detector_name} on {image_path.name} ---")

    try:
        # Load image
        image = load_image(str(image_path))
        print(f"Image loaded: {image.shape}")

        # Run detection
        result = detector.extract(image)
        print(f"Detection result: {json.dumps(result, indent=2)}")

        return True

    except Exception as e:
        print(f"Error during {detector_name} detection: {e}")
        return False

def test_missing_models():
    """Test behavior when models are missing"""
    print("\n=== Testing Missing Model Handling ===")

    try:
        # This should work and show warnings
        stamp_detector = DealerStampExtractor("models/stamp_yolo.pt")
        signature_detector = DealerSignatureExtractor("models/signature_yolo.pt")
        print("+ Detectors initialized successfully (models missing but handled gracefully)")

        # Test with a dummy image
        dummy_image = cv2.imread("data/train/172427893_3_pg11.png")
        if dummy_image is not None:
            stamp_result = stamp_detector.extract(dummy_image)
            sig_result = signature_detector.extract(dummy_image)
            print(f"+ Stamp detection result: {stamp_result}")
            print(f"+ Signature detection result: {sig_result}")
        else:
            print("- Could not load test image")

        return True

    except Exception as e:
        print(f"- Error in missing model test: {e}")
        return False

def main():
    print("YOLO Model Testing for IDFC Document Processing")
    print("=" * 50)

    # Test missing model handling
    missing_test_passed = test_missing_models()

    # Get training images
    training_images = get_training_images()
    print(f"\nFound {len(training_images)} training images for testing")

    if not training_images:
        print("No training images found. Skipping individual image tests.")
        return

    # Initialize detectors (will show warnings for missing models)
    try:
        stamp_detector = DealerStampExtractor("models/stamp_yolo.pt")
        signature_detector = DealerSignatureExtractor("models/signature_yolo.pt")
        print("+ YOLO detectors initialized")
    except Exception as e:
        print(f"- Failed to initialize detectors: {e}")
        return

    # Test on sample images
    successful_tests = 0
    total_tests = 0

    for image_path in training_images:
        print(f"\nTesting image: {image_path.name}")

        # Test stamp detection
        total_tests += 1
        if test_yolo_detection(image_path, "Stamp Detector", stamp_detector):
            successful_tests += 1

        # Test signature detection
        total_tests += 1
        if test_yolo_detection(image_path, "Signature Detector", signature_detector):
            successful_tests += 1

    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    print(f"Missing model handling: {'PASS' if missing_test_passed else 'FAIL'}")
    print(f"Individual image tests: {successful_tests}/{total_tests} passed")

    if successful_tests == total_tests:
        print("\n*** ALL YOLO TESTS PASSED ***")
        print("Note: Models are missing, so detections return default values.")
        print("This demonstrates proper error handling for missing model files.")
    else:
        print(f"\n*** {total_tests - successful_tests} TEST(S) FAILED ***")

    print("\nRecommendations:")
    print("- Obtain YOLO model files (stamp_yolo.pt, signature_yolo.pt) for full functionality")
    print("- Models should be trained on document images with stamp/signature annotations")
    print("- Current implementation gracefully handles missing models")

if __name__ == "__main__":
    main()
