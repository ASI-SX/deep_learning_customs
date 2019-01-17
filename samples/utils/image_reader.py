from PIL import Image
import numpy as np
import os
import cv2


class ImageReader():
    is_opencv_flag = True

    def __init__(self, datasets_root_path="", is_opencv = False):
        super(ImageReader, self).__init__()
        self.datasets_root_path = datasets_root_path
        self.is_opencv_flag = is_opencv

    def get_train_path_list(self):
        datasets_root_path = self.datasets_root_path
        train_path_list = []
        for index, folder in enumerate(os.listdir(datasets_root_path)):
            for filename in os.listdir(datasets_root_path + "/" + folder):
                train_path_list.append([folder + "/" + filename, index, folder])
        return train_path_list

    # def get_batch_image(self, path_list, target=(224, 224), rescale=None, batch_size=10, ont_hot_size=None, debug_flag=False):
    #     path_list = np.asarray(path_list)
    #     total_size = path_list.shape[0]
    #     batch_mask = np.random.choice(total_size, batch_size)
    #     batch_list = path_list[batch_mask]
    #     x_train = []
    #     y_train = []
    #     y_label_name = []
    #     for index, batch_item in enumerate(batch_list):
    #         img_array = self.image_generator(batch_item[0], target, rescale, debug_flag=debug_flag)
    #         x_train.append(img_array)
    #         y_train.append(batch_item[1])
    #         y_label_name.append(batch_item[2])
    #     x_train = np.asarray(x_train)
    #     y_train = np.asarray(y_train, dtype=np.int)
    #     if ont_hot_size is not None:
    #         y_train = np.identity(ont_hot_size)[y_train]
    #     y_label_name = np.asarray(y_label_name)
    #     return x_train, y_train, y_label_name

    def image_generator(self, full_path, target_h=224, target_w=224, rescale=None, shear_range=None, zoom_range=None,
                        horizontal_flip=False, debug_flag=False):

        if self.is_opencv_flag:
            # Use Opencv
            image_bgr = cv2.imread(full_path, 1)
            height, width = image_bgr.shape[0], image_bgr.shape[1]

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_cv = cv2.resize(image_rgb, (target_h, target_w))
            image_array = image_cv

            if debug_flag:
                print(image_bgr)
                cv2.imshow("image", image_bgr)

        else:
            # Use PIL
            image_pil = Image.open(full_path)
            height, width = image_pil.height, image_pil.width

            image_rgb = image_pil.convert("RGB")
            image_resize = image_rgb.resize((target_h, target_w), Image.LANCZOS)
            image_array = np.asarray(image_resize, dtype="uint8")

            if debug_flag:
                image_pil.show()

        resize_rate_h = height / target_h
        resize_rate_w = width / target_w

        if rescale is not None:
            image_array = image_array / rescale

        return image_array, resize_rate_h, resize_rate_w

