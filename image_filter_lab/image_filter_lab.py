"""
Image Filter Lab

This script uses OpenCV to apply grayscale, blur, edge detection, histogram, and pixel translation to an image.
We create a sample image and save all results to files.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create a sample image with text-like patterns
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)  # White rectangle
    cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(image, (100, 100), (300, 100), (255, 0, 0), 5)  # Blue line

    print("Image Filter Lab")
    print("=" * 20)

    # Save original
    cv2.imwrite('image_filter_lab/original_image.jpg', image)
    print("Original image saved as 'image_filter_lab/original_image.jpg'")

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('image_filter_lab/grayscale_image.jpg', gray)
    print("Grayscale image saved as 'image_filter_lab/grayscale_image.jpg'")
    print("Grayscale: Converted to single channel, removes color information.")

    # 2. Blur
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite('image_filter_lab/blurred_image.jpg', blurred)
    print("Blurred image saved as 'image_filter_lab/blurred_image.jpg'")
    print("Blur: Reduces noise and detail using Gaussian filter.")

    # 3. Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite('image_filter_lab/edges_image.jpg', edges)
    print("Edges image saved as 'image_filter_lab/edges_image.jpg'")
    print("Edge Detection: Uses Canny algorithm to find edges in the image.")

    # 4. Histogram
    plt.figure()
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title('Grayscale Histogram')
    plt.savefig('image_filter_lab/histogram.png')
    plt.close()
    print("Histogram saved as 'image_filter_lab/histogram.png'")
    print("Histogram: Shows the distribution of pixel intensities.")

    # 5. Pixel Translation
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 30]])  # Translate by 50 pixels right, 30 pixels down
    translated = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite('image_filter_lab/translated_image.jpg', translated)
    print("Translated image saved as 'image_filter_lab/translated_image.jpg'")
    print("Pixel Translation: Shifts the entire image by specified pixel amounts.")

    # 6. Sample additional image
    sample_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite('image_filter_lab/sample_image.jpg', sample_image)
    print("Sample random image saved as 'image_filter_lab/sample_image.jpg'")

    print("\nAll processed images and histogram saved. Open the files to see the results.")

if __name__ == "__main__":
    main()