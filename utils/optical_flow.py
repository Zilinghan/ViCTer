"""
Code for testing scene-change detection algorithm using optical flow.
"""
import cv2
import argparse

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help="path to video file")
args = parser.parse_args()

# Open the video file
cap = cv2.VideoCapture(args.video_path)

# resize the image
NEW_SIZE = (400, 240)
# Define threshold value for scene change detection
THRESHOLD = 8
# Initialize variables
prev_frame = None
prev_pts = None
frame_count = 0

# Create a Farneback optical flow object
flow = cv2.FarnebackOpticalFlow_create(numIters=3, winSize=3)

# Loop through all the frames in the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the new size
    frame = cv2.resize(frame, NEW_SIZE)
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow between the current frame and the previous frame
    if prev_frame is not None:
        flow_vectors = flow.calc(prev_frame, gray_frame, prev_pts)
        
        # Calculate the magnitude of the flow vectors
        mag, _ = cv2.cartToPolar(flow_vectors[...,0], flow_vectors[...,1])
        mean_mag = mag.mean()
        
        # Check if the flow magnitude is higher than the threshold
        if mean_mag > THRESHOLD:
            # Draw a message on the frame
            cv2.putText(frame, f'Scene change detected (Frame {frame_count})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw the frame index on the frame
    cv2.putText(frame, f'Frame {frame_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the current frame and wait for a key press
    cv2.imshow('Video', frame)
    
    # If the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Update the previous frame and points for optical flow calculation
    prev_frame = gray_frame
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)
    
    # Increment the frame count
    frame_count += 1

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
