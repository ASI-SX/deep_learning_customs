import numpy as np
import random
from PIL import Image
import cv2
import json
import os
import xml.etree.ElementTree as ET


class AnnotationDatasetTool():
    category_class = ['Bicycle', 'Bus', 'Car', 'Motorbike', 'Pedestrian', 'SVehicle', 'Signal', 'Signs', 'Train',
                      'Truck']
    category_class_extend = ['day', 'morning', 'night']

    def __init__(self, training_flag=False, data_list=None, label_list=None, repeat=False, one_hot_classes=None,
                 resize_flag=False, target_h=None, target_w=None, box_format_trans=None, category_class=None,
                 label_file_type=None, format="NHWC", **kwargs):
        super(AnnotationDatasetTool, self).__init__()
        self.training = training_flag
        self.data_list = data_list
        self.label_list = label_list
        self.one_hot_classes = one_hot_classes
        self.repeat = repeat
        self.total_size = data_list.shape[0]
        self.index_list = list(np.arange(0, self.total_size))
        self.resize_flag = resize_flag
        self.target_h = target_h
        self.target_w = target_w
        self.label_file_type = label_file_type
        self.format = format

        if category_class is not None:
            self.category_class = category_class
        else:
            pass
        np.random.shuffle(self.index_list)

        if training_flag:
            try:
                assert data_list.shape[0] == label_list.shape[0], "学習データサイズが異なる, 入力データサイズ = {0} ラベルサイズ = {1}".format(
                    data_list.shape[0], label_list.shape[0])
            except AssertionError as err:
                print("Error : ", err)

    def next_batch(self, batch_size):
        try:
            assert batch_size < self.total_size, "batch_sizeがtotal_batchに超えている。"
        except AssertionError as err:
            print("Error : ", err)
        if self.repeat:
            batch_mask = random.choices(self.index_list, k=batch_size)
        else:
            batch_mask = random.sample(self.index_list, k=batch_size)

        if self.training:
            x_path_batch = self.data_list[batch_mask]
            y_path_batch = self.label_list[batch_mask]
            x_batch, y_batch = self.get_train_batch(x_path_batch, y_path_batch, one_hot_classes=self.one_hot_classes)
            return x_batch, y_batch, x_path_batch, y_path_batch
        else:
            x_path_batch = self.data_list[batch_mask]
            x_batch = self.get_infer_batch(x_path_batch)
            return x_batch, x_path_batch

    def next_batch_once(self, batch_size):
        if len(self.index_list) == 0:
            self.index_reset()
        batch_mask = self.index_list[:batch_size]
        self.index_list = self.index_list[batch_size:]

        if self.training:
            x_path_batch = self.data_list[batch_mask]
            y_path_batch = self.label_list[batch_mask]
            x_batch, y_batch = self.get_train_batch(x_path_batch, y_path_batch, one_hot_classes=self.one_hot_classes)
            return x_batch, y_batch, x_path_batch, y_path_batch
        else:
            x_path_batch = self.data_list[batch_mask]
            x_batch = self.get_infer_batch(x_path_batch)
            return x_batch, x_path_batch

    def index_reset(self):
        self.index_list = list(np.arange(0, self.total_size))

    def get_train_batch(self, data_path_list, label_path_list, rescale=None, one_hot_classes=None):
        x_train = []
        y_train = []
        for index, data_path in enumerate(data_path_list):
            img_array, resize_rate_h, resize_rate_w = self.image_generator(data_path_list[index],
                                                                           target_h=self.target_h,
                                                                           target_w=self.target_w,
                                                                           rescale=rescale, format=self.format)
            train_label = self.get_annotation_label(label_path_list[index],
                                                    label_file_type=self.label_file_type,
                                                    resize_rate_h=resize_rate_h,
                                                    resize_rate_w=resize_rate_w,
                                                    one_hot_classes=one_hot_classes)
            x_train.append(img_array)
            y_train.append(train_label)
        return x_train, y_train

    def get_infer_batch(self, data_path_list, rescale=None):
        x_infer = []
        for index, data_path in enumerate(data_path_list):
            img_valid, resize_rate_h, resize_rate_w = self.image_generator(data_path, target_h=self.target_h,
                                                                           target_w=self.target_w, rescale=rescale, format=self.format)
            x_infer.append(img_valid)
        return x_infer

    def image_generator(self, full_path, target_h=224, target_w=224, rescale=None, is_opencv=True, format="NHWC",
                        histogram=True):

        if is_opencv:
            # Use Opencv
            image_bgr = cv2.imread(full_path, 1)
            height, width = image_bgr.shape[0], image_bgr.shape[1]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_cv = cv2.resize(image_rgb, (target_w, target_h))
            image_array = image_cv

            # print(image_bgr)
            # cv2.imshow("image", image_bgr)

        else:
            # Use PIL
            image_pil = Image.open(full_path)
            height, width = image_pil.height, image_pil.width

            image_rgb = image_pil.convert("RGB")
            image_resize = image_rgb.resize((target_h, target_w), Image.LANCZOS)
            image_array = np.asarray(image_resize, dtype="uint8")

            # image_pil.show()

        resize_rate_h = target_h / height
        resize_rate_w = target_w / width

        if rescale is not None:
            image_array = image_array / rescale

        if format == "NCHW":
            image_array = np.transpose(image_array, (2, 0, 1))

        return image_array, resize_rate_h, resize_rate_w

    def get_annotation_label(self, full_path, annotation_type=None, label_file_type=None, format_trans=None,
                             resize_rate_h=None, resize_rate_w=None, one_hot_classes=None):

        if label_file_type == "json":
            bbox_classes, day_time = self.get_bbox_json_ext(json_file_path=full_path, resize_rate_h=resize_rate_h,
                                                            resize_rate_w=resize_rate_w,
                                                            one_hot_classes=one_hot_classes)
            return bbox_classes

        elif label_file_type == "voc_xml":
            class_list = self.get_bbox_xml_ext(full_path, resize_rate_h=resize_rate_h,
                                               resize_rate_w=resize_rate_w,
                                               one_hot_classes=one_hot_classes)

            return class_list
        else:
            return None
            pass

    def get_bbox_json_ext(self, json_file_path, resize_rate_h=None, resize_rate_w=None, one_hot_classes=None):
        with open(json_file_path, encoding="utf-8") as json_file:
            json_data = json.loads(json_file.read())
        classes_truth = []
        ground_truth = []
        day_time = self.category_class_extend.index(json_data['attributes']['timeofday'])
        labels = json_data['labels']
        for label in labels:
            classes_truth.append(self.category_class.index(label['category']))
            ground_truth.append(
                [label['box2d']['x1'], label['box2d']['y1'],
                 label['box2d']['x2'], label['box2d']['y2']])
        classes_array = np.asarray(classes_truth)
        if one_hot_classes is not None:
            classes_array = np.identity(one_hot_classes)[classes_array]
        else:
            classes_array = np.expand_dims(classes_array, 1)

        bbox_array = np.array(ground_truth, dtype=np.float32)
        if resize_rate_h is not None:
            bbox_array[:, 1::2] *= resize_rate_h
        else:
            pass
        if resize_rate_w is not None:
            bbox_array[:, 0::2] *= resize_rate_w
        else:
            pass
        train_label = np.hstack((bbox_array, classes_array)).astype(np.uint16)

        return train_label, day_time

    def get_bbox_xml_ext(self, xml_file, resize_rate_h=None, resize_rate_w=None, one_hot_classes=None):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        train_labels = []

        for boxes in root.iter('object'):
            class_name = boxes.find('name').text
            class_id = self.category_class.index(class_name)
            if one_hot_classes is not None:
                classes_array = np.identity(one_hot_classes)[class_id]
            else:
                classes_array = np.expand_dims(class_id, 1)
            ymin, xmin, ymax, xmax = None, None, None, None

            for box in boxes.findall("bndbox"):
                xmin = int(box.find("xmin").text) * resize_rate_w
                ymin = int(box.find("ymin").text) * resize_rate_h
                xmax = int(box.find("xmax").text) * resize_rate_w
                ymax = int(box.find("ymax").text) * resize_rate_h

            boxes_item = [xmin, ymin, xmax, ymax]
            train_label = np.hstack((boxes_item, classes_array)).astype(np.uint16)
            train_labels.append(train_label)
        return train_labels

    def json_reader(self, json_file_path: str):
        with open(json_file_path, encoding="utf-8") as json_file:
            json_data = json.loads(json_file.read())
        return json_data
