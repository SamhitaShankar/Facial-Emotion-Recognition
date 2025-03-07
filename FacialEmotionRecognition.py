import cv2
import imutils
from fer import FER
import time


cam = cv2.VideoCapture(0)  


detector = FER()


area_threshold = 500
emotion_timeout = 5  


last_emotion_time = time.time()
emotion_detected = False

while True:
    
    ret, img = cam.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    
    img = imutils.resize(img, width=500)
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    
    emotion_detected = False
    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        face = img[y:y+h, x:x+w]
        
        
        emotions = detector.detect_emotions(face)
        
        if emotions:
            
            emotion, score = detector.top_emotion(face)
            
            
            cv2.putText(img, f"{emotion} ({score:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            
            emotion_detected = True
    
    
    if emotion_detected:
        last_emotion_time = time.time()
    
    
    if not emotion_detected and (time.time() - last_emotion_time) > emotion_timeout:
        
        pass
    
    
    text = "No Emotion Detected"
    if emotion_detected:
        text = "Emotion Detected"
    
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    cv2.imshow("Emotion Detection", img)
    
    
    key = cv2.waitKey(10)
    if key == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
