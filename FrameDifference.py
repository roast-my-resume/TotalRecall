import cv2
import numpy as np

# define input here
def FrameDifference(video_path, THRESHOLD = 0.5):
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    important_timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate frame difference
        diff = cv2.absdiff(prev_gray, gray)

        # calculate intensity
        intensity = np.sum(diff) / (frame.shape[0] * frame.shape[1])  # avg

        # print(intensity)
        if intensity > THRESHOLD:

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # timestamp
            important_timestamps.append(timestamp)

        prev_gray = gray

    cap.release()

    # output timestamps
    for x in important_timestamps:
        print("{:10.4f}".format(x))
