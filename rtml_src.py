import gradio as gr
import cv2
import numpy as np
from deepface import DeepFace

# Define deepface model
model = DeepFace.build_model('Emotion')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to perform emotion detection on a single frame
def detect_emotion(frame):
    try:
        if frame is not None:
            # Resize frame
            resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Preprocess the image for DeepFace
            img = gray_frame.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            # Predict emotion using DeepFace
            preds = model.predict(img)
            emotion_idx = np.argmax(preds)
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (0, 0), (200, 30), (0, 0, 0), -1)
            cv2.putText(frame, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame
    except Exception as e:
        print(f"Error: {str(e)}")
        return frame

# Create a Gradio interface
webcam = gr.inputs.Video()
output_video = gr.outputs.Video()

iface = gr.Interface(fn=detect_emotion, inputs=webcam, outputs=output_video, live=True)
iface.launch(share=True)
