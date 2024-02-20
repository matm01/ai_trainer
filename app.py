import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.title("handy dandy Trainer AI :muscle:")


def main():
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_pose = mp.solutions.pose  # Mediapipe Solutions - Pose
    st.set_page_config(page_title="Trainer AI App")
    st.title("handy dandy Trainer AI :muscle:")
    # st.caption("Powered by OpenCV, Streamlit")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not stop_button_pressed:
            ret, image = cap.read()

            if not ret:
                st.write("Video Capture Ended")
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
            frame_placeholder.image(image, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    st.button("Start", on_click=main)
