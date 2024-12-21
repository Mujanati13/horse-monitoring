from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import json
import requests

SERVER_URL = "http://84.247.166.36:3000/video-frame"

# Load the model
model = YOLO('./best_v9_yolo11s_60epoches.pt')
video_path = 'rtsp://test1234:12345678@196.135.200.43:8080/stream2'  # Replace with your public IP
# Open the video file or access a camera feed
# video_path = './test8.mp4'  # Replace with your video file path, or use 0 for webcam feed
cap = cv2.VideoCapture(video_path) # 

# Get video properties for saving the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the final output video
output_path = './outputtest8.mp4'  # Specify the output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set up Matplotlib for interactive display
plt.ion()
fig, ax = plt.subplots()
img_display = ax.imshow([[0]])  # Initialize with a blank image
plt.axis("off")

frame_count = 0
frame_skip = 5  # Set to 1 to process every frame, higher values to skip more

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frame is returned (end of video)

    # Skip frames if frame_skip is greater than 1
    if frame_count % frame_skip == 0:
        # Run YOLO inference on the frame
        results = model(frame)

        # Check if there are any predictions
        if results and len(results) > 0:
            # Extract predictions and format them as JSON
            predictions = []
            for pred in results[0].boxes:  # results[0] holds the frame's prediction data
                prediction = {
                    'class': int(pred.cls),                 # Predicted class index
                    'label': results[0].names[int(pred.cls)],  # Class label
                    'confidence': float(pred.conf),         # Confidence score
                    'box': [float(coord) for coord in pred.xywh.tolist()[0]]  # Bounding box coordinates (x, y, w, h)
                }
                predictions.append(prediction)

            # Print the JSON formatted predictions
            print(json.dumps(predictions, indent=2))

            # Extract the annotated frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video file
            out.write(annotated_frame)

            # Convert BGR (OpenCV format) to RGB for display with Matplotlib
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Update Matplotlib display with the new frame
            img_display.set_data(rgb_frame)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # **Send the frame to the server** (Optimized Sending)
            _, encoded_frame = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # try:
            #     response = requests.post(
            #         SERVER_URL,
            #         data=encoded_frame.tobytes(),
            #         headers={'Content-Type': 'image/jpeg'}
            #     )
            #     if response.status_code != 200:
            #         print(f"Error sending frame: {response.status_code}")
            # except Exception as e:
            #     print(f"Error sending frame: {e}")

    frame_count += 1

# Release resources
cap.release()
out.release()
plt.ioff()
plt.close()
