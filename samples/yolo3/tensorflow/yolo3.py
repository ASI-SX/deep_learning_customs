import tensorflow as tf
import samples.utils.tensorflow_yolo_ext as yolo_helper


class YOLOv3():
    def __init__(self, height=608, width=608, channels=3, num_classes=80, output_op_name="yolov3", lr=1e-3, momentum=0.9,
                 decay=0.0005, angle=0,
                 saturation=1.5, exposure=1.5, hue=0.1, rho1=0.9, rho2=0.999, alpha=0.1, policy="steps"):
        super(YOLOv3, self).__init__()
        self.input_shape = [None, height, width, channels]
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.output_shape = [None]
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
        self.policy = policy
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.category_class = ['Bicycle', 'Bus', 'Car', 'Motorbike', 'Pedestrian', 'SVehicle', 'Signal', 'Signs',
                               'Train', 'Truck']
        self.category_class_extend = ['day', 'morning', 'night']
        yolo3_graph = tf.Graph()

        with yolo3_graph.as_default():
            inputs = tf.placeholder(tf.float32, self.input_shape, name="inputs")
            train_labels = tf.placeholder(tf.float32, self.output_shape, name="labels")
            layer_list = []
            z = yolo_helper.conv2d_input(inputs, filter=32, layer_list=layer_list)
            z = yolo_helper.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(1):
                z = yolo_helper.residual_block(z, layer_list=layer_list)
            z = yolo_helper.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(2):
                z = yolo_helper.residual_block(z, layer_list=layer_list)
            z = yolo_helper.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(8):
                z = yolo_helper.residual_block(z, layer_list=layer_list)
            mid_1 = z
            z = yolo_helper.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(8):
                z = yolo_helper.residual_block(z, layer_list=layer_list)
            mid_2 = z
            z = yolo_helper.conv2d_before_residual(z, layer_list=layer_list)
            for i in range(4):
                z = yolo_helper.residual_block(z, layer_list=layer_list)
            mid_3 = z

            block_5l_1 = yolo_helper.conv2d_5l_block(mid_3, layer_list=layer_list)
            pre_out_l1 = yolo_helper.conv2d_3x3_up(block_5l_1, layer_list=layer_list)
            feature_out_l1 = yolo_helper.feature_out(pre_out_l1, num_classes, layer_list=layer_list, name="feature_1x")

            pre_up_l1 = yolo_helper.conv2d_1x1_down(block_5l_1, layer_list=layer_list)
            up_1 = yolo_helper.up_sampling2d(pre_up_l1, layer_list=layer_list)
            route_1 = yolo_helper.route2d(up_1, mid_2, layer_list=layer_list)

            block_5l_2 = yolo_helper.conv2d_5l_block(route_1, filter=256, layer_list=layer_list)
            pre_out_l2 = yolo_helper.conv2d_3x3_up(block_5l_2)
            feature_out_l2 = yolo_helper.feature_out(pre_out_l2, num_classes, layer_list=layer_list, name="feature_2x")

            pre_up_l2 = yolo_helper.conv2d_3x3_up(block_5l_2, layer_list=layer_list)
            up_2 = yolo_helper.up_sampling2d(pre_up_l2, layer_list=layer_list)

            route_2 = yolo_helper.route2d(mid_1, up_2, layer_list=layer_list)
            block_5l_3 = yolo_helper.conv2d_5l_block(route_2, filter=128, layer_list=layer_list)
            pre_out_l3 = yolo_helper.conv2d_3x3_up(block_5l_3, layer_list=layer_list)
            feature_out_l3 = yolo_helper.feature_out(pre_out_l3, num_classes, layer_list=layer_list, name="feature_3x")

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            pb_saver = tf.train
            init = tf.global_variables_initializer()








        self.model = yolo3_graph
        self.inputs = inputs
        self.scale_1 = feature_out_l1
        self.scale_2 = feature_out_l2
        self.scale_3 = feature_out_l3
        self.train_labels = train_labels
        self.init = init
        self.saver = saver
        self.pb_saver = pb_saver
