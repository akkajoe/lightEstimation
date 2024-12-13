import cv2

image = cv2.imread('lise-in-a-white-shawl-pierre-auguste-renoir.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Haar Cascade for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
    eyes = sorted(eyes, key=lambda x: x[1])[:2]  # Keep top two eyes by position

    for (ex, ey, ew, eh) in eyes:
        if ew > 0.2 * w and eh > 0.2 * h:  # Only process reasonably sized detections
            eye = roi_color[ey:ey+eh, ex:ex+ew]
            cv2.imshow('Eye', eye)
            cv2.imwrite(f'eye_{ex}_{ey}.jpg', eye)
