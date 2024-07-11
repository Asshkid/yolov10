from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

result = {}
# Load the YOLOv8 model
model = YOLO("yolov10n.pt")

# Open the video file
video_path = "mixkit-traffic-light-directing-traffic-4272-hd-ready.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        result[frame_nmr] = {}
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        print(result.keys())

#         # Get the boxes and track IDs
#         boxes = results[0].boxes.xywh.cpu()
#         track_ids = results[0].boxes.id.int().cpu().tolist()

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Plot the tracks
#         for box, track_id in zip(boxes, track_ids):
#             x, y, w, h = box
#             track = track_history[track_id]
#             track.append((float(x), float(y)))  # x, y center point
#             if len(track) > 30:  # retain 90 tracks for 90 frames
#                 track.pop(0)

#             # Draw the tracking lines
#             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# write_csv(results, 'test.csv')
# cap.release()
# cv2.destroyAllWindows()