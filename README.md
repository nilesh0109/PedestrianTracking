# Pedestrian Detection and Tracking
This project implements a pedestrian tracker from a side-view camera.

- For object detection in each frame, pretrained YOLO-v3(on MS-COCO dataset) is used. The number of detections are reduced to keep only person class and filter out others.
- For tracking the detected objects, a kalman filter based SORT(Simple Online and Realtime Tracking) algorithm is used https://arxiv.org/abs/1602.00763

<b>Code Walkthrough</b>
- main.py  ----> main file for the detect_and_track function
- sort.py   -----> SORT algorithm implementation
- YOLO   ------> YOLO-v3 related config files(icluding pretrained-weights)
- utils.py  -------> utility methods 


<b>How to run the code</b>
```
pip install -r requirements.txt
python main.py
```

This would return the list of tracks. In order to see the whole trajectory, one need to plot the returned points on the video frames and assemble them in a video using `cv2.VideoWriter`.

# References
- Simple Online and Realtime Tracking https://arxiv.org/abs/1602.00763
- SORT implementation https://github.com/abewley/sort
- YOlOv3 An Incremental Improvement Paper: https://arxiv.org/abs/1804.02767


