from typing import List
import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
from PIL import Image


def get_inverse_pespective(perspective_matrix: np.array)-> np.array:
  """
  This method calculates the inverse of prespective matrix by homography. 
  - Takes 4 random points on the floor plane(destination_plane) and calculates the corresponding points 
  on the camera image plane(src_plane) using perspective matrix.
  - Calculates the Homography matrix to map any point in image plane to floor plane.

  Parameters
  ----------
  perspective_matrix: 3 x 4 camera prespective matrix to convert 3d homogeneous world coordinates to 
  2d homogeneous camera coordinates.

  Returns
  ----------
  3x3 homography matrix for moving from 2d homogeneous image plane to world floor plane(at z=0)
  
  """
  #Take 5 homogenous points on the floor(Unit is in Meters)
  pts_dst = np.array([[0,0,0,1],
                      [0,1,0,1],
                      [1,0,0,1],
                      [1,1,0,1],
                      [0,0,0,1]
                    ])
  #Obtain respective homogenous points on the image plane
  pts_src = (perspective_matrix @ pts_dst.T).T
  
  #convert homogenous coordinates to cartesian coorndinates
  pts_src_cart = np.array([[x/w, y/w] for x,y,w in pts_src])
  pts_dst_cart = np.array([[x/w, y/w] for x,y,z,w in pts_dst])
  
  #find the 3x3 Homography Matrix for transforming image plane to floor plane
  h, status = cv2.findHomography(pts_src_cart, pts_dst_cart)
  return h


def project_to_floor(image_coordinates: List[int], h: np.array) -> List[int]: 
  """
  This method takes the Homography matrix and the 2d image cartesian coordinates. It returns the (x, y)
  cartesian coordinates in 3d cartesian world coordinates on floor plane(at z=0). Notice that z coordinate is omitted
  here and added inside the tracking funtion. 
  
  Parameters
  ----------
  image_coordinates: 2d pixel coordinates (x,y)
  h: 3x3 Homography matrix np.array[3x3]

  Returns
  ----------
  floor_coordinates: List of x, y coordinates in 3d world of same pixel on floor plane i.e. (x,y,z) Considering z=0 and 
  ommitted here.
  """
  #adding 1 for homogenous coordinate system
  x, y, w = h @ np.array([[*image_coordinates, 1]]).T
  return [x/w, y/w]


def get_corner_coordinates(bbox: List[int]) -> List[int]:
  """
  Parameters
  ----------
  List of [cx, cy, w, h]
  cx: x-center of the bounding box
  cy: y-center of the bounding box
  w: width of the bounding box
  h: height of the bounding box
  
  Returns
  ----------
  List of bbox coordinates in [x1, y1, x2, y2] form.
  """
  cx, cy, w, h = bbox
  return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]


def plot_detections(img: Image, detections: List) -> None:
  """
  Plot the detections onto the video frame and save it.
  Parameters
  ----------
  img: Pillow Image
  Detections: List of Detections from Yolo-v3
  
  Returns
  ----------
  None
  """
  cmap = plt.get_cmap('tab20b') 
  colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
  unique_labels = detections[:, -1].cpu().unique()
  n_cls_preds = len(unique_labels)
  bbox_colors = random.sample(colors, n_cls_preds)
    
  plt.figure()
  fig, ax = plt.subplots(1, figsize=(12,9))
  ax.imshow(img)

  for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    box_h = y2 - y1
    box_w = x2 - x1
    color = bbox_colors[int(np.where(
            unique_labels == int(cls_pred))[0])]
    bbox = patches.Rectangle((x1, y1), box_w, box_h,
             linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(bbox)
    plt.text(x1, y1, s=cls_pred, 
                color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
  plt.axis('off')
  # save image
  img_path = img.filename
  plt.savefig(img_path.replace(".jpg", "-det.jpg"),        
                  bbox_inches='tight', pad_inches=0.0)
  plt.show()


