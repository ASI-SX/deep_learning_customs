import numpy as np
from copy import deepcopy

def bbox_to_anbox(bbox):
    anbox = deepcopy(bbox)
    anbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    anbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    anbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    anbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return anbox


def anbox_to_bbox(anbox):
    bbox = deepcopy(anbox)
    bbox[..., 0] = anbox[..., 0] - anbox[..., 2] / 2
    bbox[..., 2] = anbox[..., 0] + anbox[..., 2] / 2
    bbox[..., 1] = anbox[..., 1] - anbox[..., 3] / 2
    bbox[..., 3] = anbox[..., 1] + anbox[..., 3] / 2
    return bbox
