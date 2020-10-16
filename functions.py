import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BoundBox:
  def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
  # define the attributes
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax
    self.objness = objness
    self.classes = classes
    
def _sigmoid(x):
  return 1. / (1. + np.exp(-x))

#------------------------decode_netout-----------------------------------------------
# This function returns the bounding boxes which :
#1. sigmoid(Pc)> obj_thresh
#2. set the class scores(sigmoid(Pc) x sigmoid(ci)) to zero for the class scores less than obj_thresh
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):  
  # netout:(grid size,grid size,255)
  # anchors is a list of height and width of three anchor boxes (6 elements).
  grid_h, grid_w = netout.shape[:2]                                              
  nb_box = 3                                                                   
  netout = netout.reshape((grid_h, grid_w, nb_box, -1))                        # netout:(grid size,grid size,3,85)
  nb_class = netout.shape[-1] - 5                                              # nb_class=80                  
  boxes = []
  netout[:,:,:, :2]  = _sigmoid(netout[:,:,:, :2])                             # Calculate the sigmoid of tx,ty
  netout[:,:,:, 4:]  = _sigmoid(netout[:,:,:, 4:])                             # Calculate the sigmoid of Pc,C1,C2,...,C80
  netout[:,:,:, 5:]  = np.expand_dims(netout[:,:,:, 4],axis=-1) * netout[:,:,:, 5:]  # Sigmoid(Pc) x [sigmoid(C1),...sigmoid(C80)]
  netout[:,:,:, 5:] *= netout[:,:,:, 5:] > obj_thresh                          # set the class score = sigmoid(Pc) x sigmoid (Ci) to 0 if it is less than a threshold 
                                                                                     
  for i in range(grid_h*grid_w):                                                
    row = i / grid_w                                                           
    col = i % grid_w
    for b in range(nb_box):     
      #objectness= sigmoid(Pc) for the box b at grid (int(row), int(col))                                                
      objectness = netout[int(row)][int(col)][b][4]                            
      if( objectness >= obj_thresh):   
        # x_rel_cell,y_rel_cell,tw,th= sigmoid(tx),sigmoid(ty),tw,th
        x_rel_cell, y_rel_cell, tw, th = netout[int(row)][int(col)][b][:4]     
        # x_rel_cell,y_rel_cell: position of the bounding box center with respect the grid cell
        # x_rel_cell,y_rel_cell in [0,1]
        # x,y: relative position of the bounding box center with respect to the 
        #top left corner of the input(grid size,grid size) 
        x = (col + x_rel_cell) / grid_w    
        y = (row + y_rel_cell) / grid_h 
        # w:the relative width of the bounding box to the width of the image(416) 
        # h:the relative height of the bounding box to the height of the image(416)    
        w = anchors[2 * b + 0] * np.exp(tw) / net_w   
        h = anchors[2 * b + 1] * np.exp(th) / net_h         
        classes = netout[int(row)][col][b][5:]  
        # Bounding box coordinates with respect to the unit image                                
        box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)         
        boxes.append(box)
  return boxes
#---------------------correct_yolo_boxes---------------------------------------
#As we have the bounding box coordinates with resoect to the unit image, in order 
#to get the real coordinates, relative positions should be multiplied by real width and height of the image.
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):   
# image_h,image_y: real dimensions, net_w,nwt_h=416,416               
  for i in range(len(boxes)):                                                                
    boxes[i].xmin = int((boxes[i].xmin) * image_w)                              
    boxes[i].xmax = int((boxes[i].xmax) * image_w)
    boxes[i].ymin = int((boxes[i].ymin) * image_h)
    boxes[i].ymax = int((boxes[i].ymax) * image_h)
#-------------------------_interval_overlap------------------------------------
# This function is used to compute the Intersection between two intervals
def _interval_overlap(interval_a, interval_b):
  x1, x2 = interval_a
  x3, x4 = interval_b
  if x3 < x1:
    if x4 < x1:
      return 0
    else:
      return min(x2,x4) - x1
  else:
    if x2 < x3:
       return 0
    else:
      return min(x2,x4) - x3
#----------------------bbox_iou------------------------------------------------
def bbox_iou(box1, box2):
  intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
  intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
  intersect = intersect_w * intersect_h
  w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
  w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
  union = w1*h1 + w2*h2 - intersect
  return float(intersect) / union
#----------------------Non max suppression------------------------------------
# This function computes the IoU between the candidate bounding boxes and if it
# is greater than a threshold, it ignores the bounding box with less class score.
def do_nms(boxes, nms_thresh):
# boxes:object, all the candidate bounding boxes
  if len(boxes) > 0:                                                            
    nb_class = len(boxes[0].classes)                                           
  else:
    return
  for c in range(nb_class):                                                    # step 1
    sorted_indices = np.argsort([-box.classes[c] for box in boxes])            # step 2 
    for i in range(len(sorted_indices)):                                       #len(sorted_indices)=number of bounding boxes
      index_i = sorted_indices[i]                                              # Step 3, Start with the sorted_indices[0]( index of class with the highest pc.ci)
      if boxes[index_i].classes[c] !=0:                                        # Step 4, If the most probable bounding box (that surrounds object of class c) has pc.ci!=0
          for j in range(i+1, len(sorted_indices)):                            # Step 5, 1st iteration:range(1,number of bounding boxes)
              index_j = sorted_indices[j]                                             
              if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:       # Step 6
                  boxes[index_j].classes[c] = 0                                # Set the class(ci) of the box to zero if (IoU>nms_thresh)
# We do not remove the whole bounding box, as it might have another class with  
#(ci)>obj_thresh in "classes" vector.
                  
#--------------------get_boxes-------------------------------------------------
#This function keeps the bounding boxes with class score higher than a threshold
# and returns the boxes, their respective class scores and labels.
# Each bounding box can only be labeled as a single class.
# In this code one bounding box can have only one label.
def get_boxes(boxes, labels, thresh):
  v_boxes, v_labels, v_scores = [], [], []
  score_max=0
  # iterate over the candidate boxes(resulted from the non max suprression)
  for box in boxes:                                                             
    # enumerate all possible labels
    for i in range(len(labels)):
      # check if the threshold for this label is high enough
      if box.classes[i] > thresh:  
        # Check if the class score for this class is geater than the maximum class score in this bounding box so far                                             
        if box.classes[i]*100 > score_max:                                      
          label_max=labels[i]
          score_max= box.classes[i]*100
      #exclude the box in which all the classes are 0 
      if ((i==len(labels)-1) and (score_max!=0)):                              
        v_boxes.append(box)
        # return only the label of the class with the max class score
        v_labels.append(label_max) 
        # return only maximum class score                                             
        v_scores.append(score_max)                                              
        score_max=0   
        
  # Modification: remove the bounding boxes with very high IoU and different class labels                                                                               
  index_drop=[]
  for i in range(len(v_boxes)):
    for j in range(i+1,len(v_boxes)):
      iou=bbox_iou(v_boxes[i], v_boxes[j])
      min_probability=min(v_scores[i],v_scores[j])
      if ((iou>=0.8) and (v_labels[i]!=v_labels[j])):
        index_drop.append(v_scores.index(min_probability))

  v_boxes=[v_boxes[i] for j, i in enumerate(range(len(v_boxes))) if j not in set(index_drop)]
  v_scores=[v_scores[i] for j, i in enumerate(range(len(v_scores))) if j not in set(index_drop)]
  v_labels=[v_labels[i] for j, i in enumerate(range(len(v_labels))) if j not in set(index_drop)]  
  return v_boxes, v_labels, v_scores
#-------------------------draw_boxes-------------------------------------------
def draw_boxes(filename, v_boxes, v_labels, v_scores):                          # final boxes
  # load the image
  data = pyplot.imread(filename)
  # plot the image
  pyplot.figure(figsize=(15,10))
  pyplot.imshow(data)
  # get the context for drawing boxes
  ax = pyplot.gca()                                                             # get the axes of the current image
  # plot each box
  for i in range(len(v_boxes)):
    box = v_boxes[i]
    # get coordinates
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='green')        #create object rect, (x1,y1): the bottom and left corner coordinate
    # draw the box
    ax.add_patch(rect)
    pyplot.text(x1, y1, "{} {}".format(v_labels[i], v_scores[i]), color='white') # add a text to a specific point on the image
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False,labelleft=False)
  # show the plot
  pyplot.show()  
  