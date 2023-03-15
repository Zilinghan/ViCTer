"""
annotate_video.py
    - Let the user enter the path to the video to anotate.
"""

import cv2
import argparse

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help="path to video file")
args = parser.parse_args()

# Open the video file
cap = cv2.VideoCapture(args.video_path)

# Set up the initial frame counter
frame_index = 0

# Loop through each frame in the video
while True:
    # Read in the next frame
    ret, frame = cap.read()

    # If we've reached the end of the video, exit the loop
    if not ret:
        break

    # Display the frame index on the frame
    cv2.putText(frame, f"Frame {frame_index}, press 'd' for next frame, 'a' for previous frame or 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the current frame
    cv2.imshow('Frame', frame)

    # Wait for the user to press a key
    key = cv2.waitKey(0)

    # If the user pressed 'q', exit the loop
    if key == ord('q'):
        break

    # If the user pressed the left arrow key, go back one frame
    elif key == ord('a'):
        # Decrement the frame index if it's greater than 0
        if frame_index > 0:
            frame_index -= 1

        # Set the current frame position to the previous frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # If the user pressed the right arrow key, go forward one frame
    elif key == ord('d'):
        # Increment the frame index if it's less than the total number of frames
        if frame_index < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            frame_index += 1

        # Set the current frame position to the next frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    else: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
