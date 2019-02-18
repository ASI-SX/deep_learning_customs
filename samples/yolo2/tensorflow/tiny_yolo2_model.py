import tensorflow as tf
import numpy as np
import samples.utils.tensorflow_yolo_ext as yolo_helper
from samples.utils.box_convert import bbox_to_anbox


class TINY_YOLO_v2():
    anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]

    def __init__(self, height=416, width=416, channels=3, num_classes=80, output_op_name="output", lr=1e-3,
                 momentum=0.9, ignore_thresh=0.5, decay=0.0005, is_train=False, format="NCHW", batch_size=10):
        super(TINY_YOLO_v2, self).__init__()
        if format == "NCHW":
            self.input_shape = [None, channels, height, width]
        elif format == "NHWC":
            self.input_shape = [None, height, width, channels]
        else:
            self.input_shape = [None]
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.output_size = (num_classes + 5) * 5
        self.output_shape = [None, (num_classes + 5) * 5]
        self.output_op_name = output_op_name
        self.block_size = 32
        self.grid_w = width // self.block_size
        self.grid_h = height // self.block_size
        self.batch_size = batch_size

        self.COORD = 1.0
        self.OBJ = 5.0
        self.NO_OBJ = 1.0
        self.CLASS = 1.0

        self.threshold = 0.5  # The threshold of the probability of the classes
        self.ignore_thresh = ignore_thresh
        self.num_anchor = len(self.anchors)

        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        tiny_yolo2_graph = tf.Graph()

        with tiny_yolo2_graph.as_default():
            inputs = tf.placeholder(tf.float32, self.input_shape, name="inputs")
            train_labels = tf.placeholder(tf.float32, self.output_shape, name="labels")
            conv_1 = yolo_helper.darknetconv2d(inputs, output_size=16, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_1 = yolo_helper.darknetpool(conv_1, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_2 = yolo_helper.darknetconv2d(pool_1, output_size=32, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_2 = yolo_helper.darknetpool(conv_2, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_3 = yolo_helper.darknetconv2d(pool_2, output_size=64, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_3 = yolo_helper.darknetpool(conv_3, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_4 = yolo_helper.darknetconv2d(pool_3, output_size=128, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_4 = yolo_helper.darknetpool(conv_4, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_5 = yolo_helper.darknetconv2d(pool_4, output_size=256, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_5 = yolo_helper.darknetpool(conv_5, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_6 = yolo_helper.darknetconv2d(pool_5, output_size=512, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_6 = yolo_helper.darknetpool(conv_6, h_kernel=2, w_kernel=2, h_stride=1, w_stride=1, name="maxpool")
            conv_7 = yolo_helper.darknetconv2d(pool_6, output_size=1024, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            conv_8 = yolo_helper.darknetconv2d(conv_7, output_size=512, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            output = yolo_helper.darknetconv2d(conv_8, output_size=self.output_size, h_kernel=1, w_kernel=1, h_stride=1,
                                               w_stride=1, activation="linear", batch_normalization=True,
                                               name=self.output_op_name)

            if is_train:
                with tf.name_scope("train"):
                    y_true = tf.placeholder(tf.float32,
                                            [None, self.grid_w, self.grid_h, self.num_anchor, self.num_classes + 5],
                                            name="y_true")
                    y_mask = tf.placeholder(tf.float32, [None, self.grid_w, self.grid_h, self.num_anchor, 1],
                                            name="obj_mask")

                    feature = tf.transpose(output, perm=[0, 3, 2, 1])
                    total_loss, object_loss, coord_loss, class_loss = self.loss(feature, y_true, y_mask)

                    # tf.losses.add_loss(class_loss)
                    # tf.losses.add_loss(object_loss)
                    tf.losses.add_loss(total_loss)

                    self.class_loss = class_loss
                    self.object_loss = object_loss
                    self.coord_loss = coord_loss
                    self.global_step = tf.train.create_global_step()
                    self.total_loss = tf.losses.get_total_loss()
                    self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, 100000, 0.96,
                                                                    staircase=True)
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                momentum=self.momentum)
                    self.train_op = self.optimizer.minimize(loss=self.total_loss)
                    self.y_true = y_true
                    self.y_mask = y_mask

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            pb_saver = tf.train
            init = tf.global_variables_initializer()

        self.graph = tiny_yolo2_graph
        self.inputs = inputs
        self.output = output
        self.train_labels = train_labels
        self.init = init
        self.saver = saver
        self.pb_saver = pb_saver

    def loss(self, y_pred, y_true, y_mask):
        # true box mask shape = (-1, 13, 13, 5)
        object_mask = y_mask
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        ## model output data shape = (-1, 13. 13. 125)

        ## data reshape = (-1, 13. 13. 5, 25)
        y_pred = tf.reshape(y_pred, [-1, self.grid_w, self.grid_h, self.num_anchor, self.num_classes + 5])
        ## get grid shape = (13, 13)
        xy_offset = yolo_helper.get_offset(grid_w=self.grid_w, grid_h=self.grid_h)

        ## split the xy, wh, conf, classes
        xy_pred, wh_pred, conf_pred, boxes_classes_pred = tf.split(y_pred, [2, 2, 1, self.num_classes], axis=-1)
        ## xy shape = (-1, 13, 13, 5, 2) sigmoid + offset
        # (ex: grid[2][2] = (sigmoid([-0.2, 0.3]) + [2, 2] ) * block_size 32
        boxes_xy_pred = (tf.nn.sigmoid(xy_pred) + xy_offset) * self.block_size

        # (ex: grid[2][2] = (exp([0.1, 0.2]) * anchors (shape=(5,2))) * block_size 32
        boxes_wh_pred = (tf.exp(wh_pred) * self.anchors) * self.block_size

        # concat xy and wh
        boxes_pred = tf.concat([boxes_xy_pred, boxes_wh_pred], axis=-1)

        # conf
        conf_pred = tf.nn.sigmoid(conf_pred)

        # classes
        classes_pred = object_mask * boxes_classes_pred

        ## training data , shape = (-1, 13, 13, 5, 25)
        boxes_true, conf_true, classes_true = tf.split(y_true, [4, 1, self.num_classes], axis=-1)

        xy_true = boxes_true[..., 0:2] / 32 - xy_offset
        wh_true = tf.log(boxes_true[..., 2:4] / self.anchors + 1e-7)

        iou = self.box_iou(boxes_pred, boxes_true)
        iou_scores = object_mask * iou
        best_ious = tf.reduce_max(iou_scores, axis=-1)

        conf_pred_delta = conf_pred * tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), axis=-1)
        obj_delta = tf.square((conf_pred - conf_pred_delta)) * object_mask * self.OBJ
        noobj_delta = tf.square((conf_pred - 0)) * noobject_mask * self.NO_OBJ
        full_obj_delta = obj_delta + noobj_delta

        xy_delta = (tf.square(object_mask * (xy_pred - xy_true)))
        wh_delta = (tf.square(object_mask * (wh_pred - wh_true)))

        obj_loss = tf.reduce_mean(full_obj_delta)
        xy_loss = tf.reduce_mean(xy_delta) * self.COORD
        wh_loss = tf.reduce_mean(wh_delta) * self.COORD
        coord_loss = xy_loss + wh_loss
        class_loss = tf.reduce_sum(
            tf.square(tf.nn.softmax_cross_entropy_with_logits_v2(labels=classes_true, logits=classes_pred)))

        total_loss = obj_loss + coord_loss + class_loss

        return total_loss, obj_loss, coord_loss, class_loss

    def box_iou(self, boxes1, boxes2):
        b1_xy = boxes1[..., :2]
        b1_wh = boxes1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]

        b2_xy = boxes2[..., :2]
        b2_wh = boxes2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]

        intersect_mins = tf.maximum(b1_mins, b2_mins)
        intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        iou = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-iou')
        iou = tf.expand_dims(iou, -1)
        return iou

    def pre_process_true_boxes(self, label_list, grid_w, grid_h):
        block_h = int(self.height / grid_h)
        block_w = int(self.width / grid_w)
        y_true = []
        y_mask = []
        for labels in label_list:
            labels_container = np.zeros([grid_w, grid_h, self.num_anchor, self.num_classes + 5], dtype="float32")
            labels_mask = np.zeros([grid_w, grid_h, self.num_anchor, 1], dtype="uint8")
            for label in labels:
                bbox = label[0:4]
                classes = label[4:]
                anbox = bbox_to_anbox(bbox)  # [x1 y1 x2 y2] to [x y w h]
                anchors_area = np.asarray(self.anchors)[..., 0] / np.asarray(self.anchors)[..., 1]
                # print(anchors_area)
                box_area = anbox[2] / anbox[3]
                # print(box_area)

                iou = np.abs(anchors_area - box_area)
                # print(iou)

                i = anbox[0] // block_w
                j = anbox[1] // block_h
                k = np.argmin(iou)
                # print(i, j, k)
                labels_container[i, j, k, 0:4] = anbox
                labels_container[i, j, k, 4] = 1
                labels_container[i, j, k, 5:] = classes
                labels_mask[i, j, :] = 1
            y_true.append(labels_container)
            y_mask.append(labels_mask)
        return y_true, y_mask

    def get_offset(self, grid_w, grid_h):
        grid_x = np.arange(grid_h)
        grid_y = np.arange(grid_w)
        x, y = np.meshgrid(grid_y, grid_x)
        x = np.reshape(x, (grid_w, grid_h, -1))
        y = np.reshape(y, (grid_w, grid_h, -1))
        x_y_offset = np.concatenate((x, y), -1)
        x_y_offset = np.reshape(x_y_offset, [grid_w, grid_h, 1, 2])
        return x_y_offset
