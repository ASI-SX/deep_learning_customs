import cv2
import numpy as np
import colorsys
from PIL import ImageFont, ImageDraw


def draw_boxes(image, boxes, scores, labels, classes, detection_size,
               font='./font/FiraMono-Medium.otf', show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """

    if boxes is None:
        pass
    else:
        draw = ImageDraw.Draw(image)
        # draw settings
        font = ImageFont.truetype(font=font, size=np.floor(2e-2 * image.size[1]).astype('int32'))
        hsv_tuples = [(x / len(classes), 0.9, 1.0) for x in range(len(classes))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        for i in range(len(labels)):
            # for each bounding box, do:
            bbox, score, label = boxes[i], scores[i], classes[labels[i]]
            bbox_text = "%s %.2f" % (label, score)
            text_size = draw.textsize(bbox_text, font)
            # convert_to_original_size
            detection_size, original_size = np.array(detection_size), np.array(image.size)
            ratio = original_size / detection_size
            bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))

            draw.rectangle(bbox, outline=colors[labels[i]], width=3)
            text_origin = bbox[:2] - np.array([0, text_size[1]])
            draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], fill=colors[labels[i]])
            # # draw bbox
            draw.text(tuple(text_origin), bbox_text, fill=(0, 0, 0), font=font)

        image.show() if show else None
        return image


# def read_coco_names(class_file_name):
#     names = {}
#     with open(class_file_name, 'r') as data:
#         for ID, name in enumerate(data):
#             names[ID] = name.strip('\n')
#     return names


'''--------Test the scale--------'''
if __name__ == "__main__":
    draw_boxes()