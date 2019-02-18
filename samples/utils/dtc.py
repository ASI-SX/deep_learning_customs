
import numpy as np

class TrueData(object):
    def __init__(self, original_size=None, scaled_size=None):
        self.x1 = []
        self.x2 = []
        self.y1 = []
        self.y2 = []
        self.category =[]
        self.original_size= original_size
        self.scaled_size = scaled_size

    CATEGORY_IDS = dict()
    MAX_ID = 0

    
    
    @classmethod
    def conv_category_id(cls, category):
        if cls.CATEGORY_IDS.get(category) is None:
            cls.MAX_ID +=1
            cls.CATEGORY_IDS.update({category:cls.MAX_ID})
            return cls.MAX_ID
        else:
            return cls.CATEGORY_IDS[category]

    def append(self, x1, x2, y1, y2, category):

        self.x1.append(x1)
        self.x2.append(x2)
        self.y1.append(y1)
        self.y2.append(y2)
        self.category.append(category)

    def get_number_of_boxes_in_image(self):
        return len(self.x1)

    def get_labels(self):
        return np.array(self.category)

    def get_bboxes(self):
        a = []
        for x1, x2, y1, y2 in zip(self.x1, self.x2, self.y1, self.y2):

            a.append([x1, y1, x2, y2]) # order is x_min, y_min, x_max, y_max
        return np.array(a)