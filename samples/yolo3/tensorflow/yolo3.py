import tensorflow as tf
from samples.utils import tenserflow_ext as tf_helper
from samples.utils.tensorflow_yolo_ext import YoloNN


class YOLOv3():
    def __init__(self, rows=608, cols=608, channels=3, num_classes=80, output_op_name="yolo",lr=1e-3, momentum=0.9, decay=0.0005, angle=0,
                 saturation=1.5, exposure=1.5, hue=0.1, dropout=1.0, rho1=0.9, rho2=0.999, alpha=0.1, policy="steps",
                 scales=(0.1, 0.1), ):
        super(YOLOv3, self).__init__()
        self.input_shape = [None, rows, cols, channels]
        self.rows = rows
        self.cols = cols
        self.output_shape = [None, num_classes + 5]
        self.output_op_name = output_op_name

        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.angle = angle
        self.saturation = saturation
        self.exposure = exposure
        self.hue = hue
        self.rho1 = rho1
        self.rho2 = rho2
        self.alpha = alpha
        self.scales = scales
        self.policy = policy

        mlp_graph = tf.Graph()

        with mlp_graph.as_default():
            x = tf.placeholder(tf.float32, self.input_shape, name="input")
            y_t = tf.placeholder(tf.float32, self.output_shape, name="label")
            layer_list = []
            yolo_nn = YoloNN()
            z = yolo_nn.conv2d_input(x, filter=32, layer_list=layer_list)
            z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(1):
                z = yolo_nn.residual_block(z, layer_list=layer_list)
            z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(2):
                z = yolo_nn.residual_block(z, layer_list=layer_list)
            z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(8):
                z = yolo_nn.residual_block(z, layer_list=layer_list)
            mid_1 = z
            z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(8):
                z = yolo_nn.residual_block(z, layer_list=layer_list)
            mid_2 = z
            z = yolo_nn.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(4):
                z = yolo_nn.residual_block(z, layer_list=layer_list)
            mid_3 = z

            block_5l_1 = yolo_nn.conv2d_5l_block(mid_3, layer_list=layer_list)
            pre_out_l1 = yolo_nn.conv2d_3x3_up(block_5l_1, layer_list=layer_list)
            feature_out_l1 = yolo_nn.feature_out(pre_out_l1, layer_list=layer_list, name="feature_1x")

            pre_up_l1 = yolo_nn.conv2d_1x1_down(block_5l_1, layer_list=layer_list)
            up_1 = yolo_nn.up_sampling2d(pre_up_l1, layer_list=layer_list)
            route_1 = yolo_nn.route2d(up_1, mid_2, layer_list=layer_list)

            block_5l_2 = yolo_nn.conv2d_5l_block(route_1, filter=256, layer_list=layer_list)
            pre_out_l2 = yolo_nn.conv2d_3x3_up(block_5l_2)
            feature_out_l2 = yolo_nn.feature_out(pre_out_l2, layer_list=layer_list, name="feature_2x")

            pre_up_l2 = yolo_nn.conv2d_3x3_up(block_5l_2, layer_list=layer_list)
            up_2 = yolo_nn.up_sampling2d(pre_up_l2, layer_list=layer_list)

            route_2 = yolo_nn.route2d(mid_1, up_2, layer_list=layer_list)
            block_5l_3 = yolo_nn.conv2d_5l_block(route_2, filter=128, layer_list=layer_list)
            pre_out_l3 = yolo_nn.conv2d_3x3_up(block_5l_3, layer_list=layer_list)
            feature_out_l3 = yolo_nn.feature_out(pre_out_l3, layer_list=layer_list, name="feature_3x")

            yolo_nn.yolo3(feature_out_l1)


        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=1)
        pb_saver = tf.train
        init = tf.global_variables_initializer()

        # self.model = mlp_graph
        # self.train_op = train_op
        # self.predict_op = predict_op
        # self.loss_op = cross_entropy
        # self.x = x
        # self.y = y
        # self.y_t = y_t
        # self.init = init
        # self.saver = saver
        # self.pb_saver = pb_saver