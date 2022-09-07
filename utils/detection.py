from torch.functional import Tensor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np


m_config = get_cfg()

# This line should be used if run on CPU rather than GPU.
# m_config.MODEL.DEVICE = "cpu"

m_config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
m_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
m_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
DET_MODEL = DefaultPredictor(m_config)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.
 
    Many things should be taken into account to make it work:
       1. Current model being used can detect up to 80 different objects,
          we're only looking for 'cars' or 'trucks', so you should ignore
          other detected objects.
       2. The object detector may find more than one vehicle in the picture,
          you must then, choose the one with the largest area in the image.
       3. The model can also fail and detect zero objects in the picture,
          in that case, you should return coordinates that cover the full
          image, i.e. [0, 0, width, height].
       4. Coordinates values must be integers, we're making reference to
          a position in a numpy.array, we can't use float values.
 
    Parameters
    ----------
    img : numpy.ndarray
       Image in RGB format.
 
    Returns
    -------
    box_coordinates : tuple
       Tuple having bounding box coordinates as (left, top, right, bottom).
       Also known as (x1, y1, x2, y2).
    """
    outputs = DET_MODEL(img)

    instances = outputs["instances"]
    detected_class_indexes = instances.pred_classes
    prediction_boxes = instances.pred_boxes

    biggest = 0
    for idx, coordinates in enumerate(prediction_boxes):
        class_index = detected_class_indexes[idx]
        area = instances.pred_boxes[idx].area()[0].item() 

        if area > biggest and (class_index == 2 or class_index == 7):
            biggest = area
            coord = np.array(Tensor.cpu(coordinates))

    try:
        box_coordinates = [int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])] 
    except:
        box_coordinates = [0, 0, img.shape[1], img.shape[0]]

    return box_coordinates
