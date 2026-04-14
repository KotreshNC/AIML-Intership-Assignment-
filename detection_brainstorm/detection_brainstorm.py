"""
Detection Brainstorm

This script lists 5 uses of face/object detection and implements a designed solution: a smart attendance system using face detection.
"""

import cv2
import numpy as np

def list_uses():
    print("5 Uses of Face/Object Detection:")
    print("1. Security Systems: Access control and surveillance in buildings.")
    print("2. Social Media: Automatic face tagging in photos.")
    print("3. Autonomous Vehicles: Detecting pedestrians, vehicles, and obstacles.")
    print("4. Retail Analytics: Customer counting and behavior analysis.")
    print("5. Healthcare: Patient monitoring and fall detection.")

def smart_attendance_system():
    """
    Designed Solution: Smart Attendance System
    - Uses face detection to identify and count attendees in a classroom photo.
    - Draws rectangles around detected faces and saves the marked image.
    - Prints the number of attendees detected.
    """
    print("\nDesigned Solution: Smart Attendance System")
    print("-" * 40)

    # Create a sample image with multiple faces (simulated)
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    # Simulate faces as circles
    cv2.circle(image, (100, 100), 30, (255, 255, 255), -1)  # Face 1
    cv2.circle(image, (250, 120), 25, (255, 255, 255), -1)  # Face 2
    cv2.circle(image, (400, 150), 35, (255, 255, 255), -1)  # Face 3

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the result
    cv2.imwrite('detection_brainstorm/attendance_marked.jpg', image)

    print(f"Number of attendees detected: {len(faces)}")
    print("Marked image saved as 'detection_brainstorm/attendance_marked.jpg'")
    print("In a real system, this would integrate with a database to mark attendance.")

def main():
    list_uses()
    smart_attendance_system()

if __name__ == "__main__":
    main()