from typing import List, Dict
import numpy as np
import cv2
from pydarknet import Detector, Image
import sort
from utils import *


class Object_Detector():
  """
  YOLO-v3 based object detector. This YOLO-v3 is pretrained on MS-COCO dataset.
  """
  def __init__(self):
    self.network = Detector(bytes("yolo/cfg/yolov3.cfg", encoding="utf-8"), 
                            bytes("yolo/weights/yolov3.weights", encoding="utf-8"), 0,
                            bytes("yolo/cfg/coco.data",encoding="utf-8"))

  def detect(self, img: Image, category: str=None) -> List:
    """
    Parameters
    ----------
    img: PIL Input Image
    category: category of the object to filter(should be one of the categories from MS-COCO dataset)
  
    Returns
    ----------
    detections: List of detections. Each detection is a tuple of form (object_name, score, bbox).
    """
    img = Image(img)
    detections = self.network.detect(img)
    if category is not None:
      detections = self.filter_by_category(category)
    return detections
    
  def filter_by_category(self, detections: List, category: str=None) -> List:
    """
    Parameters
    ----------
    detections: List of detections, returned by yolo-v3 model
    category: Obeject Category to be kept
    
    Returns
    ----------
    filtered_detections by the provided category. Whole List is returned if category is None or empty
    """
    return filter(lambda detection: detection[0].decode("utf-8")==category, detections) if category else detections


def detect_and_track(video_filename: str) -> Dict[str, List]:
  """
  Detection and Tracking function based on YOLO-v3 object detector and kalman filter based SORT tracker.
  Parameters
    ----------
    video_frames: path to the video file. Video would be a 4 dimesional np array of shape <N, C, H, W>.
    
    Returns
    ----------
    tracks: Dictionary of tracks where each key is the objectID and value is the list of the center of the
    object on the floor plane.
  """

  tracks = {}
  person_detector = Object_Detector()
  person_tracker = sort.Sort()
  # 1. Start reading the video file frame by frame
  cap = cv2.VideoCapture(video_filename) 

  while cap.isOpened():
    # 2. Iterate through each frame in the video
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Get the detections from the object detector
    detections = person_detector.detect(img, 'person')

    # 4. Transform the detected points on floor plane from camera image plane
    detections_on_floor_plane = []
    for (obj, score, [cx,cy,w,h]) in detections:
      #convert coordinates cx,cy,w,h to x1,y1,x2,y2. Project them onto floor plane and
      # reorder the results to (bbox, score, object_name)
      x1, y1, x2, y2 = get_corner_coordinates([cx, cy, w, h])
      detection = [x1, y1, x2, y2, score]

    # 5. Find association of the detected objects and add the objects into list of tracks Using SORT.
      if detection is not None:
      # 6. Update the tracks
        tracked_persons = person_tracker.update(detection)
        for x1, y1, x2, y2, personid in tracked_persons:
  
      # 7. For each tracked object, get the center pixel on the image plane and add it to the object trajectory.
          center_pos = (int((x1 + x2)/2), int(y1 + y2)/2)
          tracks[personid] = tracks.get(personid, []) + [center_pos]
  return tracks


if __name__ == '__main__':
  video_path =   '/Videos/MOT16-13-raw.webm' #Video Frames to input
  tracks = detect_and_track(video_path)
  


