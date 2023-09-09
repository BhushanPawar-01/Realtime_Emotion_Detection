from deepface import DeepFace
import cv2
import numpy as np

# define deepface model
model = DeepFace.build_model('Emotion')

# define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# video capture
cap = cv2.VideoCapture(0)

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # resize frame
    resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)

    # convert frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # preprocess the image for DeepFace
    img = gray_frame.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # predict emotion using DeepFace
    preds = model.predict(img)
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]

    # draw rectangle around face and label with predicted emotion
    cv2.rectangle(frame, (0, 0), (200, 30), (0, 0, 0), -1)
    cv2.putText(frame, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # display the result
    cv2.imshow('Real-time Emotion Detection', frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('camera')
