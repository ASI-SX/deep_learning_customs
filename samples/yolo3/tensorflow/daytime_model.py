import tensorflow as tf
import samples.utils.tenserflow_ext as tf_helper
import samples.utils.tensorflow_yolo_ext as yolo_helper


class DayTime():
    def __init__(self, height=1216, width=1920, channels=3, num_classes=3, output_op_name="output", lr=1e-3,
                 momentum=0.9,
                 decay=0.0005, angle=0,
                 saturation=1.5, exposure=1.5, hue=0.1, rho1=0.9, rho2=0.999, alpha=0.1, policy="steps"):
        super(DayTime, self).__init__()
        self.input_shape = [None, width, height, channels]
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
        daytime_graph = tf.Graph()

        with daytime_graph.as_default():
            inputs = tf.placeholder(tf.float32, self.input_shape, name="inputs")
            train_labels = tf.placeholder(tf.int32, self.output_shape, name="labels")
            z = tf.layers.conv2d(inputs, filters=64, kernel_size=(8, 8), strides=(8, 8))
            z = tf.layers.max_pooling2d(z, pool_size=8, strides=8)
            z = tf.layers.flatten(z)
            z = tf.layers.dense(z, 1024)
            z = tf.layers.dense(z, 512)
            y = tf.layers.dense(z, num_classes, name=output_op_name)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=train_labels)
            train_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=rho1, beta2=rho2).minimize(loss)
            accuracy = y

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            pb_saver = tf.train
            init = tf.global_variables_initializer()

        self.accuracy = accuracy
        self.train_op = train_op
        self.model = daytime_graph
        self.inputs = inputs
        self.train_labels = train_labels
        self.init = init
        self.saver = saver
        self.pb_saver = pb_saver
