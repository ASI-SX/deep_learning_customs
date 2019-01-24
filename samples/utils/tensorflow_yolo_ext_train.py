from os import path
from glob import glob
import json
import tensorflow as tf
import numpy as np
import samples.utils.tenserflow_ext as tf_helper

from tensorflow.image import ResizeMethod


class YoloNN():
    _ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    # [W,H]
#     _NUM_CLASSES = 80
    _NUM_CLASSES = 10
    meta_classes = ["person", "bicycle", "car"]

    def __init__(self):
        super(YoloNN, self).__init__()
        self.feature_maps = []
        self.inputs = None
        self.graph = None
        self.lr = 1e-3
        self.batch_size = 1

    def residual_block(self, x, filter=None, format="NHWC", name="residual_block", batch_normalization=True,
                       layer_list=None):
        """
        :param x: input tensor
        :param filter: first conv2d layer filter size. if None, it will be a half of the input tensor channel size.
        :param format: "NHWC" for channel last and "NCHW" for channel first. default is 'NHWC'
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            shortcut = x
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C / 2)
            filter_2 = C

            block_conv_1 = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name="layer_1", h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_residual : ", block_conv_1.shape)
            block_conv_2 = tf_helper.add_conv2d(block_conv_1, filter_2, h_kernel=3, w_kernel=3, name="layer_2",
                                                h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_residual : ", block_conv_2.shape)
            y = tf_helper.add_shortcut(block_conv_2, shortcut, name=scope)
            print("shortcut : ", y.shape)
            if layer_list is not None:
                layer_list.append(block_conv_1)
                layer_list.append(block_conv_2)
            return y

    def conv2d_before_residual(self, x, filter=None, format="NHWC", name="conv_before_residual",
                               batch_normalization=True,
                               layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C * 2)

            y = tf_helper.add_conv2d(x, filter_1, h_kernel=3, w_kernel=3, name=scope, h_stride=2,
                                     w_stride=2, format=format, batch_normalization=batch_normalization,
                                     activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_before_residual : ", y.shape)
            if layer_list is not None:
                layer_list.append(y)
            return y

    def conv2d_input(self, x, filter=32, format="NHWC", name="conv_input", batch_normalization=True,
                     layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else C
            y = tf_helper.add_conv2d(x, filter_1, h_kernel=3, w_kernel=3, name=scope, h_stride=1,
                                     w_stride=1, format=format, batch_normalization=batch_normalization,
                                     activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_input : ", y.shape)
            if layer_list is not None:
                layer_list.append(y)
            return y

    def conv2d_1x1_down(self, x, filter=None, format="NHWC", name="conv_1x1_down", batch_normalization=True,
                        layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C / 2)

            y = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                     w_stride=1, format=format, batch_normalization=batch_normalization,
                                     activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_1x1_down : ", y.shape)
            if layer_list is not None:
                layer_list.append(y)
            return y

    def conv2d_1x1_up(self, x, filter=None, format="NHWC", name="conv_1x1_up", batch_normalization=True,
                      layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C * 2)

            y = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                     w_stride=1, format=format, batch_normalization=batch_normalization,
                                     activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_1x1_up : ", y.shape)
            if layer_list is not None:
                layer_list.append(y)
            return y

    def conv2d_3x3_down(self, x, filter=None, format="NHWC", name="conv_3x3_down", batch_normalization=True,
                        layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C / 2)

            y = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                     w_stride=1, format=format, batch_normalization=batch_normalization,
                                     activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_3x3_down : ", y.shape)
            if layer_list is not None:
                layer_list.append(y)
            return y

    def conv2d_3x3_up(self, x, filter=None, format="NHWC", name="conv_1x1_up", batch_normalization=True,
                      layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C * 2)

            y = tf_helper.add_conv2d(x, filter_1, h_kernel=3, w_kernel=3, name=scope, h_stride=1,
                                     w_stride=1, format=format, batch_normalization=batch_normalization,
                                     activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_3x3_up : ", y.shape)
            if layer_list is not None:
                layer_list.append(y)
            return y

    def feature_out(self, x, classes_num=None, format="NHWC", name="feature_out", batch_normalization=True,
                    layer_list=None):
        """

        :param x:
        :param classes_num:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        if classes_num is None:
            classes_num = self._NUM_CLASSES

        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = (classes_num + 5) * 3

            feature = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                           w_stride=1, format=format, batch_normalization=batch_normalization,
                                           activation="linear", leaky_relu_alpha=0.1, padding="SAME")
            print("conv_feature : ", feature.shape)
        if layer_list is not None:
            layer_list.append(feature)
        return feature

    def up_sampling2d(self, x, format="NHWC", name="up_sampling",
                      layer_list=None):
        """

        :param x:
        :param format:
        :param name:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
                x = tf.image.resize_images(x, [H * 2, W * 2], method=ResizeMethod.NEAREST_NEIGHBOR)

            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
                x = tf.transpose(x, perm=[0, 2, 3, 1])
                x = tf.image.resize_images(x, [H * 2, W * 2], method=ResizeMethod.NEAREST_NEIGHBOR)
                x = tf.transpose(x, perm=[0, 3, 1, 2])
            y = x
            print("up_sampling : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y

    def route2d(self, x, sub_tensor=None, route_dim=3, format="NHWC", name="route", layer_list=None):
        """

        :param x:
        :param sub_tensor:
        :param route_dim:
        :param format:
        :param name:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            if sub_tensor is not None:
                x = tf.concat([x, sub_tensor], route_dim, name=scope)
            y = x
            print("route2d : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y

    def conv2d_5l_block(self, x, filter=None, format="NHWC", name="conv_5l", batch_normalization=True,
                        layer_list=None):
        """

        :param x:
        :param filter:
        :param format:
        :param name:
        :param batch_normalization:
        :param layer_list:
        :return:
        """
        with tf.name_scope(name) as scope:
            x = tf.convert_to_tensor(x)
            input_shape = x.get_shape()
            N, H, W, C = (0, 0, 0, 0)
            if format == "NHWC":
                N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            elif format == "NCHW":
                N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

            filter_1 = filter if filter is not None else int(C / 2)
            filter_2 = int(filter_1 * 2)
            filter_3 = filter_1
            filter_4 = int(filter_1 * 2)
            filter_5 = filter_1

            block_conv_1 = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name="layer_1", h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv5l : ", block_conv_1.shape)

            block_conv_2 = tf_helper.add_conv2d(block_conv_1, filter_2, h_kernel=3, w_kernel=3, name="layer_2",
                                                h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv5l : ", block_conv_2.shape)

            block_conv_3 = tf_helper.add_conv2d(block_conv_2, filter_3, h_kernel=1, w_kernel=1, name="layer_3",
                                                h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv5l : ", block_conv_3.shape)

            block_conv_4 = tf_helper.add_conv2d(block_conv_3, filter_4, h_kernel=3, w_kernel=3, name="layer_4",
                                                h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv5l : ", block_conv_4.shape)

            block_conv_5 = tf_helper.add_conv2d(block_conv_4, filter_5, h_kernel=1, w_kernel=1, name="layer_5",
                                                h_stride=1,
                                                w_stride=1, format=format, batch_normalization=batch_normalization,
                                                activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
            print("conv5l : ", block_conv_5.shape)
            y = block_conv_5

            if layer_list is not None:
                layer_list.append(block_conv_1)
                layer_list.append(block_conv_2)
                layer_list.append(block_conv_3)
                layer_list.append(block_conv_4)
                layer_list.append(block_conv_5)
            return y

    def reorg_layer(self, feature_map, num_classes, anchors, model_h, model_w):
            num_anchors = len(anchors) # num_anchors=3
            grid_size = feature_map.shape.as_list()[1:3] # grid_size [38, 60]
            print("anchors", anchors)
            print("grid_size", grid_size)
            print("feature_map", feature_map) # Tensor("feature_1x/batch_normalization/FusedBatchNorm:0", shape=(2, 38, 60, 45), dtype=float32)
            # the downscale image in height and weight
#             stride = tf.cast(self.img_size // grid_size, tf.float32) # [h,w] -> [y,x]
            stride = tf.cast(np.array([model_h, model_w]) // grid_size, tf.float32) # [h,w] -> [y,x]
            feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])
    
            box_centers, box_sizes, conf_logits, prob_logits = tf.split(
                feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)
    
            box_centers = tf.nn.sigmoid(box_centers)
    
            grid_x = tf.range(grid_size[1], dtype=tf.int32)
            grid_y = tf.range(grid_size[0], dtype=tf.int32)
    
            a, b = tf.meshgrid(grid_x, grid_y)
            x_offset   = tf.reshape(a, (-1, 1))
            y_offset   = tf.reshape(b, (-1, 1))
            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
            x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
            x_y_offset = tf.cast(x_y_offset, tf.float32)
    
            box_centers = box_centers + x_y_offset
            box_centers = box_centers * stride[::-1]
    
            box_sizes = tf.exp(box_sizes) * anchors # anchors -> [w, h]
            boxes = tf.concat([box_centers, box_sizes], axis=-1)
            return x_y_offset, boxes, conf_logits, prob_logits


    def reorg_layer_(self, feature_out_layer, num_classes, anchors, model_h, model_w):
        x = tf.convert_to_tensor(feature_out_layer)
        input_shape = x.get_shape()

        batch, grid_h, grid_w, grid_out = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[
            3].value
        num_anchors = len(anchors)
        lout = int(grid_out / num_anchors)

        stride_h = model_h / grid_h
        print("grid_h, grid_w",grid_h, grid_w)
        print("stride_h", stride_h)
        if model_w is None:
            stride_w = model_w / grid_w

        x = tf.reshape(x, [-1, grid_h, grid_w, num_anchors, lout])
        
#         print("x", x, "num_class",num_classes) # x Tensor("train/Reshape:0", shape=(1, 38, 60, 3, 15), dtype=float32)
        feature_class, feature_xy, feature_wh, feature_conf = tf.split(x, [num_classes, 2, 2, 1], axis=-1)
        
#         print(feature_class)
#         print(feature_xy)
#         print(feature_wh)
#         print(feature_conf)

        xy_offset = self.get_offset(grid_h, grid_w)
        feature_xy = tf.nn.sigmoid(feature_xy)

        feature_xy = feature_xy + xy_offset
        self.feature_wh = feature_wh
        
        box_xy = feature_xy * stride_h
        print("feature_wh",feature_wh)
        self.feature_wh = feature_wh
        box_wh = tf.exp(feature_wh) * anchors
        self.box_wh = box_wh
        boxes = tf.concat([box_xy, box_wh], axis=-1)
        print("box_xy", box_xy)
        print("box_wh", box_wh)
        print("boxes", boxes)
        self.boxes = boxes
        return xy_offset, boxes, feature_conf, feature_class


    def get_offset_(self, grid_h, grid_w): # ToDo:use itertools.product 
        grid_x = np.arange(grid_w)
        grid_y = np.arange(grid_h)
        x, y = np.meshgrid(grid_y, grid_x)
        x = np.reshape(x, (grid_h, grid_w, -1))
        y = np.reshape(y, (grid_h, grid_w, -1))
        x_y_offset = np.concatenate((x, y), -1)
        x_y_offset = np.reshape(x_y_offset, [grid_h, grid_w, 1, 2])
        return x_y_offset

    def softmax(self, a):
        c = tf.reduce_max(a)
        exp_a = tf.exp(a - c)
        sum_exp_a = tf.reduce_sum(exp_a)
        y = exp_a / sum_exp_a
        return y
# 
#     def forward_yolo_layer(self, layer, network):
#         avg_iou = 0
#         recall = 0
#         recall75 = 0
#         avg_cat = 0
#         avg_obj = 0
#         avg_anyobj = 0
#         count = 0
#         class_count = 0
# 
#         layer.cost = 0
# 
#         return None

    def entry_index(self, layer, batch, location, entry):
        n = location / (layer.w * layer.h)
        loc = location % (layer.w * layer.h)
        out = batch * layer.outputs + n * layer.w * layer.h * (layer.classes + 4 + 1) + entry * layer.w * layer.h + loc
        return out


    def extract_feature(self, feature_map, num_classes=None, mask=[0, 1, 2], model_h=None, model_w=None, scope_name=None):
        """

        :param x: scale_1 [None, 13, 13, 3*(classes + 4 + 1)]
               or scale_2 [None, 26, 26, 3*(classes + 4 + 1)]
               or scale_3 [None, 52, 52, 3*(classes + 4 + 1)]

        :param num: 3 anchors num
        :param total: 9 all anchors num
        :param mask: anchors 3set [8, 7, 6] or [5, 4, 3] or [2, 1, 0]
        :param classes: classes def 80
        :param jitter:
        :param ignore_thresh:
        :param truth_thresh:
        :param random:
        :param max:
        :param format:
        :return: [None, 3, (classes + 4 + 1)]
        """
        if num_classes is None:
            num_classes = self._NUM_CLASSES

        
        anchors = np.array(self._ANCHORS)[mask]
        xy_offset, boxes, feature_conf, feature_class = self.reorg_layer(feature_map, num_classes, anchors, model_h, model_w)
        # xy_offset :type is ndarray
#         print(type(xy_offset),xy_offset)
#         print(type(boxes),boxes, boxes)
#         print(type(feature_conf),feature_conf)
#         print(type(feature_class),feature_class)
        return (xy_offset, boxes, feature_conf, feature_class)
    
    def non_maximum_supression(self, extracts, num_classes=None): # ToDo: グリッドないでsuppressする

        pass


    def decode_image_coord(self, num_classes=None): # ToDo: グリッド座標を画像座標に変換する

        pass


    def predict(self, extracts, num_classes=None): # ToDo: conf のしきい値を入れる
        """
        Note: given by feature_maps, compute the receptive field
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 13, 13, 255],
                                        [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        """
        
        feature_map_1, feature_map_2, feature_map_3 = extracts
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3]),]

        boxes_list, confs_list, probs_list = [], [], []

        if num_classes is None:
            num_classes = self._NUM_CLASSES

        for feature in extracts:
            (x_y_offset, boxes, confs, probs) = feature
            grid_size = x_y_offset.shape[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0]*grid_size[1]*3, 4]) # (1, 19, 19, 3, 4) => (1, 1083, 4) 
            conf_logits = tf.reshape(confs, [-1, grid_size[0]*grid_size[1]*3, 1])# (1, 19, 19, 3, 1) => (1, 1083, 1)
            prob_logits = tf.reshape(probs, [-1, grid_size[0]*grid_size[1]*3, num_classes]) # (1, 19, 19, 3, 80)  => (1, 1083, 80)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
        
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, height, width = tf.split(boxes, [1,1,1,1], axis=-1)
        x0 = center_x - height / 2
        y0 = center_y - width  / 2
        x1 = center_x + height / 2
        y1 = center_y + width  / 2

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        
        return boxes, confs, probs # ToDo: 


    def box_iou(self, box_pred, box_truth):
        """
        Calculer IoU between 2 BBs
        :param box_pred: predicted box, shape=[None, 13, 13, 3, 4], 4: xywh
        :param box_truth: true box, shape=[None, 13, 13, 3, 4], 4: xywh
        :return: iou: intersection of 2 BBs, tensor, shape=[None, 13, 13, 3] , 3: 3scalar IOU
        """
        with tf.name_scope('PredBox'):
            """Calculate 2 corners: {left bottom, right top} and area of this box"""
            b1_xy = box_pred[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
            b1_wh = box_pred[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
            b1_wh_half = b1_wh / 2.  # w/2, h/2 shape= (None, 13, 13, 3, 1, 2)
            b1_mins = b1_xy - b1_wh_half  # x,y: left bottom corner of BB
            b1_maxes = b1_xy + b1_wh_half  # x,y: right top corner of BB
            b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # w1 * h1 (None, 13, 13, 3, 1)

        with tf.name_scope('TruthBox'):
            """Calculate 2 corners: {left bottom, right top} and area of this box"""
            # b2 = tf.expand_dims(b2, -2)  # shape= (None, 13, 13, 3, 1, 4)
            # b2 = tf.expand_dims(b2, 0)  # shape= (1, None, 13, 13, 3, 4)
            b2_xy = box_truth[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
            b2_wh = box_truth[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
            b2_wh_half = b2_wh / 2.  # w/2, h/2 shape=(None, 13, 13, 3, 1, 2)
            b2_mins = b2_xy - b2_wh_half  # x,y: left bottom corner of BB
            b2_maxes = b2_xy + b2_wh_half  # x,y: right top corner of BB
            b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # w2 * h2

        with tf.name_scope('Intersection'):
            """Calculate 2 corners: {left bottom, right top} based on BB1, BB2 and area of this box"""
            # intersect_mins = tf.maximum(b1_mins, b2_mins, name='left_bottom')  # (None, 13, 13, 3, 1, 2)
            intersect_mins = tf.maximum(b1_mins, b2_mins)  # (None, 13, 13, 3, 1, 2)
            # intersect_maxes = tf.minimum(b1_maxes, b2_maxes, name='right_top')  #
            intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
            # intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)  # (None, 13, 13, 3, 1, 2), 2: w,h
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # intersection: wi * hi (None, 13, 13, 3, 1)

        IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')  # (None, 13, 13, 3, 1)
        return IoU
    

    def feature_mapping(self, bboX1, bboX2, bboxY1, bboxY2, class_category):
        """
            x1:
            x2:
            y1:
            y2:
            "category": "Pedestrian"
        """

        res = []
        
        return res

    @classmethod    
    def conv_true_feature_map(cls, true_boxes, true_labels, input_shape, anchors, num_classes):
        """
        Preprocess true boxes to training input format
        Parameters:
        -----------
        :param true_boxes: numpy.ndarray of shape [T, 4]
                            T: the number of boxes in each image.
                            4: coordinate => x_min, y_min, x_max, y_max
        :param true_labels: class id
        :param input_shape: the shape of input image to the yolov3 network, [416, 416]
        :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height
        :param num_classes: integer, for coco dataset, it is 80
        Returns:
        ----------
        y_true: list(3 array), shape like yolo_outputs, [13, 13, 3, 85]
                               13:cell szie, 3:number of anchors
                               85: box_centers, box_sizes, confidence, probability
        """
        input_shape = np.array(input_shape, dtype=np.int32)
        num_layers = len(anchors) // 3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        grid_sizes = [input_shape//32, input_shape//16, input_shape//8]
        box_centers = (true_boxes[:, 0:2] + true_boxes[:, 2:4]) / 2 # the center of box
        box_sizes =  true_boxes[:, 2:4] - true_boxes[:, 0:2] # the height and width of box
    
        true_boxes[:, 0:2] = box_centers
        true_boxes[:, 2:4] = box_sizes
    
        y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+num_classes], dtype=np.float32)
    
        y_true = [y_true_13, y_true_26, y_true_52]
    
        anchors_max =  anchors / 2.
        anchors_min = -anchors_max
        valid_mask = box_sizes[:, 0] > 0
    
    
        # Discard zero rows.
        wh = box_sizes[valid_mask]
        # set the center of all boxes as the origin of their coordinates
        # and correct their coordinates
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max
    
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
    
        anchor_area = anchors[:, 0] * anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
    
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n not in anchor_mask[l]:
                    # ToDo: Add NoObj calculation for loss when a true Object exists
                    continue
                else:
                    i = np.floor(true_boxes[t,0]/input_shape[0]*grid_sizes[l][0]).astype('int32')
                    j = np.floor(true_boxes[t,1]/input_shape[1]*grid_sizes[l][1]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_labels[t].astype('int32')
        
                    y_true[l][i, j, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][i, j, k,   4] = 1
                    y_true[l][i, j, k, 5+c] = 1
    
        return y_true_13, y_true_26, y_true_52

    def softmax(self, a):
        c = tf.reduce_max(a)
        exp_a = tf.exp(a - c)
        sum_exp_a = tf.reduce_sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    
    def loss(self, y_pred, y_true, mask, model_h=1216, model_w=1920, scope_name="",
             max_box_per_image=100, ignore_thresh=0.5): # input_size should be hold by graph (ex:tf.constant)
        """
            ToDo: move ignore_thresh to init
        """
        with tf.variable_scope("loss", reuse=True):
            anchors = np.array(self._ANCHORS)[mask]
            image_size = tf.constant([model_h, model_w])
            # size in [h, w] format! don't get messed up!
            grid_size = tf.shape(y_pred)[1:3]
            grid_size_ = y_pred.shape.as_list()[1:3]
    
            y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5+self._NUM_CLASSES])
    
            # the downscale ratio in height and weight
            ratio = tf.cast(image_size / grid_size, tf.float32)
            # N: batch_size
            N = tf.cast(tf.shape(y_pred)[0], tf.float32)
            print("y_pred", y_pred) # (2, 38, 60, 45)
            x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(y_pred, YoloNN._NUM_CLASSES, anchors, model_h, model_w)
            # shape: take 416x416 input image and 13*13 feature_map for example:
            # [N, 13, 13, 3, 1]
            object_mask = y_true[..., 4:5]
            # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
            # V: num of true gt box
            valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))
    
            # shape: [V, 2]
            valid_true_box_xy = valid_true_boxes[:, 0:2]
            valid_true_box_wh = valid_true_boxes[:, 2:4]
            # shape: [N, 13, 13, 3, 2]
            pred_box_xy = pred_boxes[..., 0:2]
            pred_box_wh = pred_boxes[..., 2:4]
    
            # calc iou
            # shape: [N, 13, 13, 3, V]
            iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
    
            # shape: [N, 13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # get_ignore_mask
            ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
            # shape: [N, 13, 13, 3, 1]
            ignore_mask = tf.expand_dims(ignore_mask, -1)
            # get xy coordinates in one cell from the feature_map
            # numerical range: 0 ~ 1
            # shape: [N, 13, 13, 3, 2]
            true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
            pred_xy = pred_box_xy      / ratio[::-1] - x_y_offset
    
            # get_tw_th, numerical range: 0 ~ 1
            # shape: [N, 13, 13, 3, 2]
            true_tw_th = y_true[..., 2:4] / anchors
            pred_tw_th = pred_box_wh      / anchors
            # for numerical stability
            true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                                  x=tf.ones_like(true_tw_th), y=true_tw_th)
            pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                                  x=tf.ones_like(pred_tw_th), y=pred_tw_th)
    
            true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
            pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))
    
            # box size punishment:
            # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
            # shape: [N, 13, 13, 3, 1]
            box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(image_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(image_size[0], tf.float32))
#             box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))
    
            # shape: [N, 13, 13, 3, 1]
            xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
            wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N
    
            # shape: [N, 13, 13, 3, 1]
            conf_pos_mask = object_mask
            conf_neg_mask = (1 - object_mask) * ignore_mask
            conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
            conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
            conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N
    
            # shape: [N, 13, 13, 3, 1]
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
            class_loss = tf.reduce_sum(class_loss) / N
    
            return xy_loss, wh_loss, conf_loss, class_loss
    
    
    def loss_(self, y_pred, y_true, mask, model_h=1216, model_w=1920, scope_name="",
             max_box_per_image=100, ignore_thresh=0.5): # input_size should be hold by graph (ex:tf.constant)
        """
            ToDo: move ignore_thresh to init
        """
        NO_OBJECT_SCALE  = 1.0
        OBJECT_SCALE     = 5.0
        COORD_SCALE      = 1.0
        CLASS_SCALE      = 1.0
        """
        image h =1216, w = 1920
        feature_maps.append(tf.convert_to_tensor(np.arange(1 * 38 * 60 * 3 * 15, dtype=np.float32).reshape((1, 38, 60, 3, 15))))
        feature_maps.append(tf.convert_to_tensor(np.arange(1 * 76 * 120 * 3 * 15, dtype=np.float32).reshape((1, 76, 120, 3, 15))))
        feature_maps.append(tf.convert_to_tensor(np.arange(1 * 152 * 240 * 3 * 15, dtype=np.float32).reshape((1, 152, 240, 3, 15))))
        """
        loss_xy=0.0
        loss_wh=0.0
        loss_obj_conf=0.0
        loss_noobj_conf=0.0
        loss_class=0.0
        recall50=0.0
        recall75=0.0
        avg_iou = 0.0

        with tf.variable_scope("loss", reuse=True):
#             image_size = tf.constant([model_w, model_h])
            image_size = tf.constant([model_h, model_w])
            anchors = np.array(self._ANCHORS)[mask]
            self.anchors = tf.constant(anchors)
            xy_offset, pred_boxes, pred_box_conf_logits, pred_box_class_logits = self.reorg_layer(y_pred, YoloNN._NUM_CLASSES, anchors, model_h, model_w)
            print("pred_boxes", pred_boxes)
            grid_size = xy_offset.shape[:2] # [13, 13] 
            stride = tf.cast(tf.constant([model_h, model_w])//grid_size, dtype=tf.float32)

            # (13, 13, 1, 2), (1, 13, 13, 3, 4), (1, 13, 13, 3, 1), (1, 13, 13, 3, 80)
            
            """
            Adjust prediction
            """
            # Tensor: Tensor("train/loss/Sigmoid_1:0", shape=(1, 38, 60, 3, 1), dtype=float32)
            pred_box_conf  = tf.nn.sigmoid(pred_box_conf_logits)                                    # adjust confidence

            # Tensor: Tensor("train/loss/ArgMax:0", shape=(1, 38, 60, 3), dtype=int64)
#             pred_box_class = tf.argmax(self.softmax(pred_box_class_logits), -1)                    # adjust class probabilities
            pred_box_class = tf.argmax(tf.nn.softmax(pred_box_class_logits), -1)                    # adjust class probabilities

            # Tensor: Tensor("train/loss/truediv:0", shape=(1, 38, 60, 3, 2), dtype=float32)
            pred_box_xy = (pred_boxes[..., 0:2] + pred_boxes[..., 2:4]) / 2.                        # absolute coordinate
            self.pred_box_xy = pred_box_xy
            # Tensor: Tensor("train/loss/sub:0", shape=(1, 38, 60, 3, 2), dtype=float32)
            pred_box_wh =  pred_boxes[..., 2:4] - pred_boxes[..., 0:2]                               # absolute size
            self.pred_box_wh = pred_box_wh

            """
            Adjust ground truth
            """
            true_box_class = tf.argmax(y_true[..., 5:], -1)
            true_box_conf  = y_true[..., 4:5]
            true_box_xy = y_true[..., 0:2]                                                           # absolute coordinate
            true_box_wh = y_true[..., 2:4]                                                           # absolute size
            object_mask = y_true[..., 4:5]
        
            # initially, drag all objectness of all boxes to 0
            conf_delta  = pred_box_conf - 0

            """
            Compute some online statistics
            """
            print("true_box_xy", true_box_xy.shape, true_box_wh.shape)
            print("pred_box_xy", pred_box_xy.shape, pred_box_wh.shape) 
            true_mins = true_box_xy - true_box_wh / 2.
            true_maxs = true_box_xy + true_box_wh / 2.
            pred_mins = pred_box_xy - pred_box_wh / 2.
            pred_maxs = pred_box_xy + pred_box_wh / 2.

            intersect_mins  = tf.maximum(pred_mins,  true_mins)
            intersect_maxs = tf.minimum(pred_maxs, true_maxs)
            

            intersect_wh    = tf.maximum(intersect_maxs - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]        

            true_area = true_box_wh[..., 0] * true_box_wh[..., 1]
            pred_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
            
            union_area  = pred_area + true_area - intersect_area
            iou_scores  = tf.truediv(intersect_area, union_area)
 
            # object_mask, intersect_area, iou_scores
            # iou_scores IOU Tensor: Tensor("train/loss/truediv_5:0", shape=(?, 38, 60, 3), dtype=float32)
            # ObjectMask Tensor: Tensor("train/loss/strided_slice_8:0", shape=(?, 38, 60, 3, 1), dtype=float32)
            # count Tensor: Tensor("train/loss/Sum:0", shape=(), dtype=float32)
            # class_mask Tensor: Tensor("train/loss/ExpandDims:0", shape=(?, 38, 60, 3, 1), dtype=float32)
            # detect_mask Tensor: Tensor("train/loss/ToFloat:0", shape=(?, 38, 60, 3, 1), dtype=float32)
            iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
            count       = tf.reduce_sum(object_mask)
            detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
            class_mask  = tf.expand_dims(tf.to_float(tf.equal(pred_box_class, true_box_class)), 4)
            print("iou_scores", iou_scores)
            print("detect_mask  * class_mask", detect_mask  * class_mask)
#             recall50    = tf.reduce_mean(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
#             recall75    = tf.reduce_mean(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)
#             avg_iou     = tf.reduce_mean(iou_scores) / (count + 1e-3)

            """
            Compare each predicted box to all true boxes
            """
            def pick_out_gt_box(y_true):
                y_true = y_true.copy()
                bs = y_true.shape[0]
                # print("=>y_true", y_true.shape)
                true_boxes_batch = np.zeros([bs, 1, 1, 1, max_box_per_image, 4], dtype=np.float32)
                # print("=>true_boxes_batch", true_boxes_batch.shape)
                for i in range(bs):
                    y_true_per_layer = y_true[i]
                    true_boxes_per_layer = y_true_per_layer[y_true_per_layer[..., 4] > 0][:, 0:4]
                    if len(true_boxes_per_layer) == 0: continue
                    true_boxes_batch[i][0][0][0][0:len(true_boxes_per_layer)] = true_boxes_per_layer
    
                return true_boxes_batch

            true_boxes = tf.py_func(pick_out_gt_box, [y_true], [tf.float32] )[0]
            self.true_boxes = true_boxes
            print("true_boxes", true_boxes)
            true_xy = true_boxes[..., 0:2]  # absolute location
            true_wh = true_boxes[..., 2:4]  # absolute size
    
    
            true_mins = true_xy - true_wh / 2.
            true_maxs = true_xy + true_wh / 2.
            pred_mins = tf.expand_dims(pred_boxes[..., 0:2], axis=4)
            pred_maxs = tf.expand_dims(pred_boxes[..., 2:4], axis=4)
            pred_wh   = pred_maxs - pred_mins
    
            intersect_mins  = tf.maximum(pred_mins, true_mins)
            intersect_maxs  = tf.minimum(pred_maxs, true_maxs)
    
            intersect_wh    = tf.maximum(intersect_maxs - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    
            true_area = true_wh[..., 0] * true_wh[..., 1]
            pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    
            union_area = pred_area + true_area - intersect_area
            iou_scores  = tf.truediv(intersect_area, union_area)
            best_ious   = tf.reduce_max(iou_scores, axis=4)
            conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)
    
            """
            Compare each true box to all anchor boxes
            """
            ### adjust x and y => relative position to the containing cell
            true_box_xy = true_box_xy / stride  - xy_offset      # t_xy  in `sigma(t_xy) + c_xy`
            pred_box_xy = pred_box_xy / stride  - xy_offset
            print("pred_box_xy" , pred_box_xy)
            ### adjust w and h => relative size to the containing cell
            true_box_wh_logit = true_box_wh / anchors
            pred_box_wh_logit = pred_box_wh / anchors
            
            true_box_wh_logit = tf.where(condition=tf.equal(true_box_wh_logit,0),
                                         x=tf.ones_like(true_box_wh_logit), y=true_box_wh_logit)
            pred_box_wh_logit = tf.where(condition=tf.equal(pred_box_wh_logit,0),
                                         x=tf.ones_like(pred_box_wh_logit), y=pred_box_wh_logit)
            print("true_box_wh_logit", true_box_wh_logit)
            print("pred_box_wh_logit", pred_box_wh_logit)
            self.true_box_wh_logit = true_box_wh_logit
            self.pred_box_wh_logit = pred_box_wh_logit
#             true_box_wh = tf.log(true_box_wh_logit)              # t_wh in `p_wh*exp(t_wh)
            # tf.clip_by_value(calculated_output, 1e-37, 1e+37)
#             true_box_wh = tf.log(tf.clip_by_value(true_box_wh_logit,1e-10,1.0))              # t_wh in `p_wh*exp(t_wh)`
#             pred_box_wh = tf.log(tf.clip_by_value(pred_box_wh_logit,1e-10,1.0))
#             true_box_wh = tf.log(tf.clip_by_value(true_box_wh_logit,1e-10,1.0))              # t_wh in `p_wh*exp(t_wh)`
#             pred_box_wh = tf.log(tf.clip_by_value(pred_box_wh_logit,1e-10,1.0))
#             pred_box_wh = tf.log(pred_box_wh_logit)
            self.true_box_wh = true_box_wh
            self.pred_box_wh = pred_box_wh
            # image_size
#             wh_scale = tf.exp(true_box_wh) * anchors / tf.to_float(self.img_size)
            wh_scale = tf.exp(true_box_wh) * anchors / tf.to_float(image_size)
            print("wh_scale", wh_scale) 
            # wh_scale Tensor("train/loss_2/truediv_13:0", shape=(?, 152, 240, 3, 2), dtype=float32)

#             self.true_box_wh = true_box_wh
#             self.pred_box_wh = pred_box_wh
            wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale
            self.wh_scale = wh_scale    
#             self.predtrue = pred_box_wh-true_box_wh
            xy_delta    = object_mask   * (pred_box_xy-true_box_xy) * wh_scale * COORD_SCALE
            wh_delta    = object_mask   * (pred_box_wh-true_box_wh) * wh_scale * COORD_SCALE # ToDo squaringze diff 
#             self.wh_delta = wh_delta
            conf_delta  = object_mask   * (pred_box_conf-true_box_conf) * OBJECT_SCALE
            conf_delta += (1-object_mask) * conf_delta * NO_OBJECT_SCALE

            class_delta = object_mask * \
                          tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class_logits), 4) * CLASS_SCALE
    
#             loss_xy    = tf.reduce_mean(tf.square(xy_delta))
#             loss_wh    = tf.reduce_mean(tf.square(wh_delta))
#             loss_conf  = tf.reduce_mean(tf.square(conf_delta))
#             loss_class = tf.reduce_mean(tf.square(class_delta))
            loss_xy    = tf.reduce_mean(tf.square(xy_delta))
            loss_wh    = tf.reduce_mean(tf.square(wh_delta))
            loss_conf  = tf.reduce_mean(tf.square(conf_delta))
            loss_class = tf.reduce_mean(tf.square(class_delta))

#             self.loss_xy = loss_xy
#             self.loss_wh = loss_wh

            # res [None, (nan, 0.2793249, 0.2793249), (nan, 0.26704398, 0.26704398), (nan, 0.26334733, 0.26334733)]
#             loss_xy    = tf.losses.mean_squared_error(true_box_xy, pred_box_xy)
#             loss_wh    = tf.losses.absolute_difference(true_box_wh, pred_box_wh,)
#             loss_conf  = tf.losses.mean_squared_error(true_box_conf, pred_box_conf)
#             loss_class = tf.losses.mean_squared_error(true_box_conf, pred_box_conf)

#         return xy_delta, wh_delta, conf_delta, class_delta
#         return loss_xy, loss_conf, loss_class
        return loss_xy + loss_wh, loss_conf, loss_class



    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area  = pred_box_wh[..., 0]  * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area  = true_box_wh[..., 0]  * true_box_wh[..., 1]
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou

def create_model(t_array, is_training=False):
    mlp_graph = tf.Graph()
    yolo_nn = None

    with mlp_graph.as_default():
        layer_list = []

        yolo_nn = YoloNN()
        yolo_nn.inputs = tf.placeholder(tf.float32, t_array.shape, name="input")
#         yolo_nn.inputs = tf.placeholder(tf.float32, t_array.shape, name="input")
        z = yolo_nn.conv2d_input(yolo_nn.inputs, filter=32, layer_list=layer_list)
        z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
        for i in range(1):
            z = yolo_nn.residual_block(z, layer_list=layer_list, name="residual_block1")
        z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
        for i in range(2):
            z = yolo_nn.residual_block(z, layer_list=layer_list,name="residual_block2")
        z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
        for i in range(8):
            z = yolo_nn.residual_block(z, layer_list=layer_list, name="residual_block8_1")
        mid_1 = z
        z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
        for i in range(8):
            z = yolo_nn.residual_block(z, layer_list=layer_list, name="residual_block8_2")
        mid_2 = z
        z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
        for i in range(4):
            z = yolo_nn.residual_block(z, layer_list=layer_list, name="residual_block8_4")
        mid_3 = z
    
        block_5l_1 = yolo_nn.conv2d_5l_block(mid_3, layer_list=layer_list)
        pre_out_l1 = yolo_nn.conv2d_3x3_up(block_5l_1, layer_list=layer_list)
        feature_1x = yolo_nn.feature_out(pre_out_l1, layer_list=layer_list, name="feature_1x")

    
        pre_up_l1 = yolo_nn.conv2d_1x1_down(block_5l_1, layer_list=layer_list)
        up_1 = yolo_nn.up_sampling2d(pre_up_l1, layer_list=layer_list)
        route_1 = yolo_nn.route2d(up_1, mid_2, layer_list=layer_list)
    
        block_5l_2 = yolo_nn.conv2d_5l_block(route_1, filter=256, layer_list=layer_list)
        pre_out_l2 = yolo_nn.conv2d_3x3_up(block_5l_2)
        feature_2x = yolo_nn.feature_out(pre_out_l2, layer_list=layer_list, name="feature_2x")
    
        pre_up_l2 = yolo_nn.conv2d_3x3_up(block_5l_2, layer_list=layer_list)
        up_2 = yolo_nn.up_sampling2d(pre_up_l2, layer_list=layer_list)
    
        route_2 = yolo_nn.route2d(mid_1, up_2, layer_list=layer_list)
        block_5l_3 = yolo_nn.conv2d_5l_block(route_2, filter=128, layer_list=layer_list)
        pre_out_l3 = yolo_nn.conv2d_3x3_up(block_5l_3, layer_list=layer_list)
        feature_3x = yolo_nn.feature_out(pre_out_l3, layer_list=layer_list, name="feature_3x")
        
        if is_training:
            with tf.name_scope("train"):
#                 loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
#                 total_loss, rec_50, rec_75,  avg_iou    = 0., 0., 0., 0.

                # WxH
#                 true_feature_map_1x = tf.placeholder(tf.float32,[None, 60, 38, 3, 15] ,"true_feature_map_1x")
#                 true_feature_map_2x = tf.placeholder(tf.float32,[None, 120, 76, 3, 15] ,"true_feature_map_2x")
#                 true_feature_map_3x = tf.placeholder(tf.float32,[None, 240, 152, 3, 15] ,"true_feature_map_3x")
#                 true_images = tf.placeholder(tf.float32, [None, ])
                # HxW
                true_feature_map_1x = tf.placeholder(tf.float32,[None, 38, 60, 3, 15] ,"true_feature_map_1x") # 
                true_feature_map_2x = tf.placeholder(tf.float32,[None, 76, 120, 3, 15] ,"true_feature_map_2x") #
                true_feature_map_3x = tf.placeholder(tf.float32,[None, 152, 240, 3, 15] ,"true_feature_map_3x") #    
#                 mask_1x_index =tf.constant([6, 7, 8])
#                 mask_2x_index =tf.constant([3, 4, 5])                
#                 mask_3x_index =tf.constant([0, 1, 2])
                mask_1x_index =[6, 7, 8]
                mask_2x_index =[3, 4, 5]                
                mask_3x_index =[0, 1, 2]
                #  xy_delta, wh_delta, conf_delta, class_delta
                loss_1x =yolo_nn.loss(feature_1x, true_feature_map_1x, mask=mask_1x_index,
                                                    model_h=1216, model_w=1920,
                                                    scope_name="loss_1x") # todo reuse scope for training                
                
                loss_2x =yolo_nn.loss(feature_2x, true_feature_map_2x, mask=mask_2x_index,
                                                    model_h=1216, model_w=1920,
                                                    scope_name="loss_2x")

                loss_3x =yolo_nn.loss(feature_3x, true_feature_map_3x, mask=mask_3x_index,
                                                    model_h=1216, model_w=1920, 
                                                    scope_name="loss_3x")

                #             loss_xy    = tf.reduce_mean(tf.square(xy_delta))
#             loss_wh    = tf.reduce_mean(tf.square(wh_delta))
#             loss_conf  = tf.reduce_mean(tf.square(conf_delta))
#             loss_class = tf.reduce_mean(class_delta)
                
                yolo_nn.coord_loss = loss_1x[0] + loss_2x[0]+ loss_3x[0]
                yolo_nn.object_loss = loss_1x[1] + loss_2x[1]+ loss_3x[1]
                yolo_nn.class_loss = loss_1x[2] + loss_2x[2]+ loss_3x[2] # class loss

                
                tf.losses.add_loss(yolo_nn.class_loss)
                tf.losses.add_loss(yolo_nn.object_loss)
                tf.losses.add_loss(yolo_nn.coord_loss)

                # tf.losses.mean_squared_error 
                yolo_nn.global_step = tf.train.create_global_step()
#                 print("global_step" , yolo_nn.global_step)
                yolo_nn.total_loss = tf.losses.get_total_loss() #(add_regularization_losses, name)
#                 print("total_loss" , yolo_nn.total_loss)
                yolo_nn.optimizer = tf.train.GradientDescentOptimizer(
                                                learning_rate=yolo_nn.lr)
                # (loss, global_step, var_list, gate_gradients, aggregation_method, colocate_gradients_with_ops, name, grad_loss)
                yolo_nn.class_train_op =  yolo_nn.optimizer.minimize(loss=yolo_nn.class_loss)
                yolo_nn.object_train_op =  yolo_nn.optimizer.minimize(loss=yolo_nn.object_loss)
                yolo_nn.coord_train_op =  yolo_nn.optimizer.minimize(loss=yolo_nn.coord_loss)
                yolo_nn.train_op =  yolo_nn.optimizer.minimize(loss=yolo_nn.total_loss, global_step=yolo_nn.global_step)

                # true feature maps(ToDo: confirm to remove or not)
                yolo_nn.true_feature_map_1x = true_feature_map_1x
                yolo_nn.true_feature_map_2x = true_feature_map_2x 
                yolo_nn.true_feature_map_3x = true_feature_map_3x
                # losses(ToDo: confirm to remove or not)
                yolo_nn.loss_1x = loss_1x
                yolo_nn.loss_2x = loss_2x 
                yolo_nn.loss_3x = loss_3x
                print("yolo_nn.loss_1x", yolo_nn.loss_1x)
                print("yolo_nn.loss_2x", yolo_nn.loss_2x)
                print("yolo_nn.loss_3x", yolo_nn.loss_3x)
    yolo_nn.graph = mlp_graph

    return yolo_nn



def preprocess_true_boxes(true_boxes, true_labels, input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format
    Parameters:
    -----------
    :param true_boxes: numpy.ndarray of shape [T, 4]
                        T: the number of boxes in each image.
                        4: coordinate => x_min, y_min, x_max, y_max
    :param true_labels: class id
    :param input_shape: the shape of input image to the yolov3 network, [416, 416]
    :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height
    :param num_classes: integer, for coco dataset, it is 80
    Returns:
    ----------
    y_true: list(3 array), shape like yolo_outputs, [13, 13, 3, 85]
                           13:cell szie, 3:number of anchors
                           85: box_centers, box_sizes, confidence, probability
    """
    input_shape = np.array(input_shape, dtype=np.int32)
    num_layers = len(anchors) // 3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    grid_sizes = [input_shape//32, input_shape//16, input_shape//8]
#     print(grid_sizes) # [array([60, 38], dtype=int32), array([120,  76], dtype=int32), array([240, 152], dtype=int32)]

    box_centers = (true_boxes[:, 0:2] + true_boxes[:, 2:4]) / 2 # the center of box
    box_sizes =  true_boxes[:, 2:4] - true_boxes[:, 0:2] # the height and width of box

    true_boxes[:, 0:2] = box_centers
    true_boxes[:, 2:4] = box_sizes

    y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+num_classes], dtype=np.float32)
    y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+num_classes], dtype=np.float32)
    y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+num_classes], dtype=np.float32)


    y_true = [y_true_13, y_true_26, y_true_52]

#     print(y_true_13.shape, y_true_26.shape, y_true_52.shape) # (60, 38, 3, 15) (120, 76, 3, 15) (240, 152, 3, 15)

    anchors_max =  anchors / 2.
    anchors_min = -anchors_max
    valid_mask = box_sizes[:, 0] > 0


    # Discard zero rows.
    wh = box_sizes[valid_mask]
    # set the center of all boxes as the origin of their coordinates
    # and correct their coordinates
    wh = np.expand_dims(wh, -2)
    boxes_max = wh / 2.
    boxes_min = -boxes_max

    intersect_mins = np.maximum(boxes_min, anchors_min)
    intersect_maxs = np.minimum(boxes_max, anchors_max)
    intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]

    anchor_area = anchors[:, 0] * anchors[:, 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)
    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
        for l in range(num_layers):
            if n not in anchor_mask[l]:
                # ToDo: Add NoObj calculation for loss when a true Object exists
                continue
            else:
                i = np.floor(true_boxes[t,0]/input_shape[0]*grid_sizes[l][0]).astype('int32')
                j = np.floor(true_boxes[t,1]/input_shape[1]*grid_sizes[l][1]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_labels[t].astype('int32')
                y_true[l][i, j, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][i, j, k,   4] = 1
                y_true[l][i, j, k, 5+c] = 1

    return y_true_13, y_true_26, y_true_52

def infer(model, infer_data):
    extracts = []
    
    with tf.Session(graph=model.graph) as sess_predict:
        sess_predict.run(tf.global_variables_initializer())
        sess_predict.run([model.feature_maps], feed_dict={model.inputs: infer_data})
        # feature_out_l1 = (1, 13, 13, 255) => anchor (116, 90), (156, 198), (373, 326)
        # feature_out_l2 = (1, 26, 26, 255) => anchor (30, 61), (62, 45), (59, 119),
        # feature_out_l3 = (1, 52, 52, 255) => anchor (10, 13), (16, 30), (33, 23),
        for i, feature in enumerate(model.feature_maps):
            mask =list(range(-3*i+6, -3*i+9)) 
#             print(mask, type(feature),feature.shape)
            extracts.append(model.extract_feature(feature, mask=mask, model_h=1216, model_w=1920))            
        
        model.predict(extracts, model._NUM_CLASSES)
        

    # ToDo: final output decode form extracts
    return extracts


np.set_printoptions(threshold=np.inf)
def train_test(model, pred_images,
               y_true_13, y_true_26,y_true_52,
               epoch=10,to_save=True, batch_size=1): # ToDo: Set Train Session
    feed_dict= {model.inputs:pred_images, 
                model.true_feature_map_1x: y_true_13,
                model.true_feature_map_2x: y_true_26, 
                model.true_feature_map_3x: y_true_52}

    max_step = epoch//batch_size
    
    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
#         sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#         sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
#         check = tf.add_check_numerics_ops()
        for step in range(max_step):
            print("step ", step)
            feed_dict= {model.inputs:pred_images,
                model.true_feature_map_1x: y_true_13,
                model.true_feature_map_2x: y_true_26, 
                model.true_feature_map_3x: y_true_52}
#             res =sess.run([model.train_op, model.true_feature_map_1x],
#             res =sess.run([model.train_op,  model.loss_1x, model.loss_2x, model.loss_3x],
#                           feed_dict=feed_dict)
            res =sess.run([model.train_op, model.loss_1x, model.loss_2x, model.loss_3x, model.total_loss],
                          feed_dict=feed_dict)
#             res =sess.run([model.train_op, model.feature_wh, model.boxes, model.wh_scale],
#                           feed_dict=feed_dict)
#             res =sess.run([model.train_op,model.wh_scale, model.true_boxes,
#                            model.true_box_wh_logit, model.pred_box_wh_logit,
#                            model.true_box_wh, model.pred_box_wh],
#                           feed_dict=feed_dict)
#             print("model.anchors", model.anchors)
#             print("model.true_box_wh", model.true_box_wh)
#             print("model.wh_delta", model.wh_delta)
#             print("model.pred_box_xy", model.pred_box_xy)
#             print("model.pred_box_wh", model.pred_box_wh)
#             if 1<step:
            print("res")
            for i, r in enumerate(res):
                print("=" *10, i)
                print(r)

#             print(res)
#             res =sess.run([model.train_op ,model.global_step, model.total_loss, model.loss_1x, model.loss_2x, model.loss_3x],
#                           feed_dict=feed_dict)
#             print("res", res)
#             print("loss_1x", model.loss_1x)
#             print("loss_2x", model.loss_2x)
#             print("loss_3x",model.loss_3x)
#             print(model.true_feature_map_2x, model.loss_1x, model.loss_2x, model.loss_3x)


'''--------Test the scale--------'''
if __name__ == "__main__":

    # preprocess true data(convert feature map)
    from samples.utils.dtc import TrueData
    from samples.utils.image_reader import ImageReader
    sample_root = path.join(path.dirname(__file__), path.pardir, "SamplesObjDetect")
    images_root = path.join(sample_root, "images")
    annotations_root = path.join(sample_root, "annotations")

    images = []
    jsons = []

    original_size = (1936, 1216)
    input_shape = (1920, 1216) # W, H
   
    for img in glob(path.abspath(path.join(sample_root,"images" , "*.jpg"))):
        images.append(img)
        jsons.append(path.abspath(path.join(annotations_root, path.basename(img).replace("jpg", "json"))))

    reader = ImageReader()
    truedata = TrueData()
    images_array = []
    y_true_13s = [] 
    y_true_26s = [] 
    y_true_52s = []

    for c, (i, j) in enumerate(zip(images, jsons)):
        if 1<c:
            continue
        image_array, resize_rate_h ,resize_rate_w = reader.image_generator(full_path=images[0],
                                                                           target_h=input_shape[1],
                                                                           target_w=input_shape[0])
        images_array.append(image_array.transpose(1,0,2))
        with open(j) as f_anno:
                annotation = json.load(f_anno)
                for anno in annotation["labels"]:
                    truedata.append(float(anno["box2d"]["x1"]/resize_rate_w),
                                    float(anno["box2d"]["x2"]/resize_rate_w),
                                    float(anno["box2d"]["y1"]/resize_rate_h),
                                    float(anno["box2d"]["y2"]/resize_rate_h),
                                    TrueData.conv_category_id(anno["category"]))

        true_boxes = truedata.get_bboxes()
        true_labels = truedata.get_labels()

        anchors = np.array(YoloNN._ANCHORS)
        num_classes = 10
        y_true_13, y_true_26, y_true_52 = preprocess_true_boxes(true_boxes, true_labels, input_shape, anchors, num_classes)
        # ToDo:change preprocess_true_box to remove transporse
        y_true_13s.append(y_true_13.transpose(1, 0, 2, 3))  #
        y_true_26s.append(y_true_26.transpose(1, 0, 2, 3))
        y_true_52s.append(y_true_52.transpose(1, 0, 2, 3))

    images_array =np.array(images_array) # (101, 1920, 1216, 3)
#     images_array = (images_array/255.).astype(np.float32)
    images_array = (images_array).astype(np.uint8)
    model =create_model(images_array, is_training=True) # ToDo: remove t_array data, pass t_array.shape
    train_test(model,pred_images=images_array,
               y_true_13=np.array(y_true_13s),
               y_true_26=np.array(y_true_26s),
               y_true_52=np.array(y_true_52s))
