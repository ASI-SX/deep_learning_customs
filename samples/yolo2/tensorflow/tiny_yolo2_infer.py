import colorsys
from os import path
from PIL import Image
import numpy as np
import tensorflow as tf
from samples.yolo2.tensorflow.tiny_yolo2_model import TINY_YOLO_v2
from samples.utils.common_tool import sigmoid
import cv2
from samples.utils.box_convert import bbox_to_anbox, anbox_to_bbox
import samples.utils.pre_process_tool as list_reader
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool


'''
Created on 2019/01/28

@author: hayashi
'''
voc2007_classes = ['chair', 'bird', 'sofa', 'bicycle', 'cat', 'motorbike', 'bus', 'boat', 'sheep', 'bottle', 'cow',
                   'person', 'horse', 'diningtable', 'pottedplant', 'aeroplane', 'car', 'train', 'dog', 'tvmonitor']


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors


def draw(image_rgb, features, classes, width_rate, height_rate):
    num_class = len(classes)
    colors = random_colors(num_class)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for box in features:
        if box[4] > 0.9:
            print(box[0:4])
            box[0:4:2] = box[0:4:2] // width_rate
            box[1:4:2] = box[1:4:2] // height_rate
            classes_index = np.argmax(box[5:])
            print(classes_index)
            label = voc2007_classes[classes_index]
            print(label)
            # color = colors[classes_index]
            print(box[0:4])
            box = np.asarray(box, dtype="int32")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(image_bgr, label, (box[0], box[1]+10), font, 0.4, (10, 200, 10), 1, cv2.LINE_AA)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_label(image_rgb, label, classes, width_rate, height_rate):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in label:
        box[0::2] = box[0::2] // width_rate
        box[1::2] = box[1::2] // height_rate
        classes_id = np.argmax(box[4:])
        label = classes[classes_id]
        cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(image_bgr, label, (box[0], box[1]+10), font, 0.4, (10, 200, 10), 1, cv2.LINE_AA)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prepare_boxes(feature, anchors, grid_h, grid_w, offset, classes, block_size):
    num_anchors = len(anchors)
    num_classes = len(classes)
    feature = np.reshape(feature, (grid_w, grid_h, num_anchors, 5 + num_classes))
    print(feature.shape)
    feature_xy, feature_wh, feature_conf, feature_classes = np.split(feature, [2, 4, 5], axis=-1)
    print(feature_conf[1][1])
    print(feature_classes[1][1][1])
    print("="*10)

    print(feature_xy[1][1])
    print("="*5)

    feature_xy = sigmoid(feature_xy)
    print(feature_xy[1][1])
    print("="*5)

    feature_xy = (feature_xy + offset) * block_size
    print(feature_xy[1][1])
    print("="*10)

    print(feature_wh[1][1])
    print("="*5)

    feature_wh = np.exp(feature_wh)
    print(feature_wh[1][1])
    print("="*5)

    feature_wh = feature_wh * anchors
    print(feature_wh[1][1])
    print("="*5)

    feature_wh = feature_wh * block_size
    print(feature_wh[1][1])
    print("="*10)

    feature_boxes = np.concatenate((feature_xy, feature_wh), axis=-1)
    print(feature_boxes[1][1])
    feature_bbox = anbox_to_bbox(feature_boxes)
    print(feature_bbox[1][1])
    feature_conf = sigmoid(feature_conf)

    print("="*10)
    print(feature_conf[1][1])
    print(feature_classes[1][1])

    feature_concat = np.concatenate((feature_bbox, feature_conf, feature_classes), axis=-1)
    feature_prepared = feature_concat.reshape((-1, 5 + num_classes))
    return feature_prepared


def infer(model, infer_data, ckpt_file, classes, batch_size):
    data_size = infer_data.total_size
    total_batch = data_size // batch_size
    for i in range(total_batch):
        batch_x, batch_y, x_path, y_path = infer_data.next_batch_once(batch_size=batch_size)

        image_data, origin_height, origin_width = image_reader(x_path[0])
        print(origin_height)
        print(origin_width)
        model_height = model.height
        model_weight = model.width
        grid_height = model.grid_h
        grid_width = model.grid_w
        offset = model.get_offset(grid_h=grid_height, grid_w=grid_width)
        block_size = model.block_size
        anchors = model.anchors
        height_resize_rate = model_height / origin_height
        width_resize_rate = model_weight / origin_width

        print(height_resize_rate)
        print(width_resize_rate)


        tf.reset_default_graph()
        with tf.Session(graph=model.graph) as sess_predict:
            model.saver.restore(sess_predict, ckpt_file)
            output = sess_predict.run(model.output, feed_dict={model.inputs: batch_x})

        feature = prepare_boxes(output[0], anchors, grid_height, grid_width, offset, classes, block_size)
        # print out label data
        draw_label(image_data, batch_y[0], classes, width_resize_rate, height_resize_rate)
        # print out infer data
        # draw(image_data, feature, classes, width_resize_rate,height_resize_rate)


    return None


def image_resize(image_bgr, target_h, target_w, rescale=None, format="NHWC", ):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_rgb, (target_w, target_h))
    image_array = image_cv

    if rescale is not None:
        image_array = image_array / rescale

    if format == "NCHW":
        image_array = np.transpose(image_array, (2, 0, 1))

    return image_array


def image_reader(full_path):
    image_bgr = cv2.imread(full_path, 1)
    height, width = image_bgr.shape[0], image_bgr.shape[1]
    return image_bgr, height, width


if __name__ == '__main__':
    voc2007_classes = ['chair', 'bird', 'sofa', 'bicycle', 'cat', 'motorbike', 'bus', 'boat', 'sheep', 'bottle', 'cow',
                       'person', 'horse', 'diningtable', 'pottedplant', 'aeroplane', 'car', 'train', 'dog', 'tvmonitor']

    num_classes = len(voc2007_classes)

    root_path = "train_data_mini/"
    data_list_path = root_path + "images"
    label_list_path = root_path + "annotations"

    data_list = np.asarray(list_reader.get_data_path_list(data_list_path)[:1])

    label_list = np.asarray(list_reader.get_data_path_list(label_list_path)[:1])

    annotation_dataset_tool = AnnotationDatasetTool(training_flag=True, data_list=data_list, label_list=label_list,
                                                    category_class=voc2007_classes, one_hot_classes=num_classes,
                                                    resize_flag=True, target_h=416, target_w=416,
                                                    label_file_type="voc_xml", format="NCHW")
    batch_size = 1
    # init model
    tiny_yolo2_model = TINY_YOLO_v2(output_op_name="output", num_classes=20, is_train=False, width=416, height=416)

    # model weights path
    ckpt_file = "model/tiny_yolo2.ckpt"
    infer(tiny_yolo2_model, annotation_dataset_tool, ckpt_file, voc2007_classes, batch_size)
