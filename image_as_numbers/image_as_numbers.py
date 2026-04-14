"""
Image as Numbers

This script demonstrates loading an image, printing its shape, pixel values, and channels, with explanations.
Since no image file is available, we create a sample RGB image using NumPy and OpenCV.
"""

import cv2
import numpy as np

def main():
    # Create a sample 100x100 RGB image
    # Red square in top-left, green in bottom-right, blue in center
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:50, :50] = [255, 0, 0]  # Red
    image[50:, 50:] = [0, 255, 0]  # Green
    image[25:75, 25:75] = [0, 0, 255]  # Blue

    print("Image as Numbers Demo")
    print("=" * 30)

    # Print shape
    print(f"Image shape: {image.shape}")
    print("Explanation: Shape is (height, width, channels). Here, 100x100 pixels with 3 color channels (RGB).")

    # Print pixel values for a small section
    print("\nPixel values (first 5x5 pixels):")
    print(image[:5, :5])
    print("Explanation: Each pixel is represented as [B, G, R] values (0-255). 0 means no color, 255 means full intensity.")

    # Print channels
    print(f"\nNumber of channels: {image.shape[2]}")
    print("Explanation: RGB images have 3 channels - Red, Green, Blue. Grayscale images have 1 channel.")

    # Show individual channels
    print("\nRed channel (first 3x3):")
    print(image[:3, :3, 2])  # R is channel 2 in OpenCV (BGR order)
    print("Green channel (first 3x3):")
    print(image[:3, :3, 1])
    print("Blue channel (first 3x3):")
    print(image[:3, :3, 0])

    print("\nNote: OpenCV uses BGR order by default, so channel 0=Blue, 1=Green, 2=Red.")

    # Save the image for reference
    cv2.imwrite('image_as_numbers/sample_image.jpg', image)
    print("\nSample image saved as 'image_as_numbers/sample_image.jpg'")

if __name__ == "__main__":
    main()