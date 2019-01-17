import tensorflow as tf
import numpy as np
import samples.utils.tenserflow_ext as tf_helper

from tensorflow.image import ResizeMethod


class YoloNN():
    _ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
#     _NUM_CLASSES = 80
    _NUM_CLASSES = 10
    meta_classes = ["person", "bicycle", "car"]

    def __init__(self):
        super(YoloNN, self).__init__()
        self.feature_maps = []
        self.inputs = None
        self.graph = None

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


    def reorg_layer(self, feature_out_layer, num_classes, anchors, model_h, model_w):

        x = tf.convert_to_tensor(feature_out_layer)
        input_shape = x.get_shape()

        batch, grid_h, grid_w, grid_out = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[
            3].value
        num_anchors = len(anchors)
        lout = int(grid_out / num_anchors)

        stride_h = model_h / grid_h
        if model_w is None:
            stride_w = model_w / grid_w

        x = tf.reshape(x, [-1, grid_h, grid_w, num_anchors, lout])

        feature_class, feature_xy, feature_wh, feature_conf = tf.split(x, [num_classes, 2, 2, 1], axis=-1)
        
#         print(feature_class)
#         print(feature_xy)
#         print(feature_wh)
#         print(feature_conf)

        xy_offset = self.get_offset(grid_h, grid_w)
        feature_xy = tf.nn.sigmoid(feature_xy)

        feature_xy = feature_xy + xy_offset
        box_xy = feature_xy * stride_h
        box_wh = tf.exp(feature_wh) * anchors
        boxes = tf.concat([box_xy, box_wh], axis=-1)

        return xy_offset, boxes, feature_conf, feature_class



    def get_offset(self, grid_h, grid_w): # ToDo:use itertools.product 
        grid_x = np.arange(grid_w)
        grid_y = np.arange(grid_h)
        x, y = np.meshgrid(grid_y, grid_x)
        x = np.reshape(x, (grid_h, grid_w, -1))
        y = np.reshape(y, (grid_h, grid_w, -1))
        x_y_offset = np.concatenate((x, y), -1)
        x_y_offset = np.reshape(x_y_offset, [grid_h, grid_w, 1, 2])
        return x_y_offset


    def forward_yolo_layer(self, layer, network):
        avg_iou = 0
        recall = 0
        recall75 = 0
        avg_cat = 0
        avg_obj = 0
        avg_anyobj = 0
        count = 0
        class_count = 0

        layer.cost = 0

        return None

    def entry_index(self, layer, batch, location, entry):
        n = location / (layer.w * layer.h)
        loc = location % (layer.w * layer.h)
        out = batch * layer.outputs + n * layer.w * layer.h * (layer.classes + 4 + 1) + entry * layer.w * layer.h + loc
        return out


    def extract_feature(self, feature_map, num_classes=None, mask=[0, 1, 2], model_h=None, model_w=None):
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
        print(anchors)


        xy_offset, boxes, feature_conf, feature_class = self.reorg_layer(feature_map, num_classes, anchors, model_h, model_w)
        # xy_offset :type is ndarray
#         print(type(xy_offset),xy_offset)
        print(type(boxes),boxes, boxes)
        print(type(feature_conf),feature_conf)
        print(type(feature_class),feature_class)
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


    def loss(self, loss):
        return tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')


def create_model(t_array, is_training=False):
    mlp_graph = tf.Graph()
    yolo_nn = None
#     init = tf.global_variables_initializer()
    with mlp_graph.as_default():
        layer_list = []

        yolo_nn = YoloNN()
        yolo_nn.inputs = tf.placeholder(tf.float32, t_array.shape, name="input")

        z = yolo_nn.conv2d_input(t_array, filter=32, layer_list=layer_list)
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
        yolo_nn.feature_maps.append(yolo_nn.feature_out(pre_out_l1, layer_list=layer_list, name="feature_1x"))
    
        pre_up_l1 = yolo_nn.conv2d_1x1_down(block_5l_1, layer_list=layer_list)
        up_1 = yolo_nn.up_sampling2d(pre_up_l1, layer_list=layer_list)
        route_1 = yolo_nn.route2d(up_1, mid_2, layer_list=layer_list)
    
        block_5l_2 = yolo_nn.conv2d_5l_block(route_1, filter=256, layer_list=layer_list)
        pre_out_l2 = yolo_nn.conv2d_3x3_up(block_5l_2)
        yolo_nn.feature_maps.append(yolo_nn.feature_out(pre_out_l2, layer_list=layer_list, name="feature_2x"))
    
        pre_up_l2 = yolo_nn.conv2d_3x3_up(block_5l_2, layer_list=layer_list)
        up_2 = yolo_nn.up_sampling2d(pre_up_l2, layer_list=layer_list)
    
        route_2 = yolo_nn.route2d(mid_1, up_2, layer_list=layer_list)
        block_5l_3 = yolo_nn.conv2d_5l_block(route_2, filter=128, layer_list=layer_list)
        pre_out_l3 = yolo_nn.conv2d_3x3_up(block_5l_3, layer_list=layer_list)
        yolo_nn.feature_maps.append(yolo_nn.feature_out(pre_out_l3, layer_list=layer_list, name="feature_3x"))

    yolo_nn.graph = mlp_graph

    return yolo_nn


def infer(model, infer_data):
    extracts = []
    
    with tf.Session(graph=model.graph) as sess_predict:
        init = tf.global_variables_initializer()
        sess_predict.run(init)
        sess_predict.run([model.feature_maps], feed_dict={model.inputs: infer_data})
        # feature_out_l1 = (1, 13, 13, 255) => anchor (116, 90), (156, 198), (373, 326)
        # feature_out_l2 = (1, 26, 26, 255) => anchor (30, 61), (62, 45), (59, 119),
        # feature_out_l3 = (1, 52, 52, 255) => anchor (10, 13), (16, 30), (33, 23),     
        for i, feature in enumerate(model.feature_maps):
            mask =list(range(-3*i+6, -3*i+9)) 
            print(mask, type(feature),feature.shape)
            extracts.append(model.extract_feature(feature, mask=mask, model_h=1216, model_w=1920))
            
        model.predict(extracts, model._NUM_CLASSES)

    # ToDo: final output decode form extracts
    return extracts

def train(model):# ToDo: Set Train Session
    pass


'''--------Test the scale--------'''
if __name__ == "__main__":
    t_array = np.arange(1 * 1216 * 1920 * 3, dtype=np.float32).reshape((1, 1216, 1920, 3)) # h , w
    model =create_model(t_array) # ToDo: remove t_array data, pass t_array.shape 
    infer(model, t_array)



#     t_array = np.arange(1 * 608 * 608 * 3, dtype=np.float32).reshape((1, 608, 608, 3)) # h , w
    # box_1 = np.ones((1, 13, 13, 3, 4), dtype=np.float32)
    # print(box_1)
    # box_1 = box_1.reshape((1, 13, 13, 3, 2, 2)) * ((1, 1), (0.1, 0.1))
    # box_1 = box_1.reshape((1, 13, 13, 3, 4))
    # print(box_1)
    # box_2 = np.ones((1, 13, 13, 3, 4), dtype=np.float32)
    # print(box_2)
    # print(t_array)


    # iou = yolo_nn.box_iou(box_1, box_2)
    # print(iou)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(iou))

