import cv2
import streamlit as st
import numpy as np
import tempfile
import cv2
from playsound import playsound
# Use this line to capture video from the webcam
cap = cv2.VideoCapture(0)

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')
# Set the title for the Streamlit app
st.title("Video Capture with OpenCV")

frame_placeholder = st.empty()

# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()

    if not ret:
        st.write("The video capture has ended.")
        break

    # You can process the frame here if needed
    # e.g., apply filters, transformations, or object detection
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
        for (x,y,w,h) in fire:
            cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            playsound('audio.mp3')
                

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        # Convert the frame from BGR to RGB format


        # Display the frame using Streamlit's st.image
        frame_placeholder.image(frame, channels="RGB")
    
    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
        break

cap.release()
cv2.destroyAllWindows()