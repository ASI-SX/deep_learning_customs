import tensorflow as tf
import numpy as np
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool
import samples.utils.pre_process_tool as list_reader
import cv2
import pickle
import pprint

import samples.utils.tensorflow_yolo_ext as yolo_helper

'''--------Test the scale--------'''
if __name__ == "__main__":

    root_path = "/home/deven/traindata/"
    data_list_path = root_path + "DTC/dtc_train_images"
    label_list_path = root_path + "DTC/dtc_train_annotations"

    data_list = np.asarray(list_reader.get_data_path_list(data_list_path))
    label_list = np.asarray(list_reader.get_data_path_list(label_list_path))
    annotation_dataset_tool = AnnotationDatasetTool(training_flag=True, data_list=data_list, label_list=label_list,
                                                    one_hot_classes=10, resize_flag=True, target_h=1216, target_w=1920)

    batch_size = 200
    total_size = annotation_dataset_tool.total_size
    range_size = int(total_size / batch_size)

    print(total_size)
    print(batch_size)
    print(range_size)

    dark_image_path_list = []
    dark_label_path_list = []
    light_image_path_list = []
    light_label_path_list = []
    data_list = open('dark_list.txt', 'w')
    light_list = open('light_list.txt', 'w')

    for i in range(range_size):
        print("range : ", i)
        train_x, train_y, x_path, y_path = annotation_dataset_tool.next_batch_once(batch_size=batch_size)
        for index, image in enumerate(train_x):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            val = hsv.T[2].flatten().mean()
            file_name = x_path[index].split("/", -1)[-1].split(".", 1)[0]
            if val > 80:
                light_image_path_list.append(file_name)
                light_list.write(x_path[index] + "\n")
                print("V:", val, "| light_path : ", x_path[index])
            else:
                dark_image_path_list.append(file_name)
                data_list.write(x_path[index] + "\n")
                print("V:", val, "| dark_path : ", x_path[index])
    light_list.close()
    data_list.close()

    with open("light_list_binary.txt", "wb") as light_fp:
        pickle.dump(light_image_path_list, light_fp)
    with open("dark_list_binary.txt", "wb") as dark_fp:
        pickle.dump(dark_image_path_list, dark_fp)

    with open("light_list_binary.txt", "rb") as light_fp:
        light_list_load = pickle.load(light_fp)
        print(len(light_list_load))
        print(light_list_load)

    with open("dark_list_binary.txt", "rb") as dark_fp:
        dark_list_load = pickle.load(dark_fp)
        print(len(dark_list_load))
        print(dark_list_load)
