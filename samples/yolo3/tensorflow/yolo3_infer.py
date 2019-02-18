import tensorflow as tf
import numpy as np
from samples.yolo3.tensorflow.yolo3 import YOLOv3
import samples.utils.tensorflow_yolo_ext as yolo_helper


def infer(model, infer_data):
    extracts = []
    with tf.Session(graph=model.model) as sess_predict:
        init = tf.global_variables_initializer()
        sess_predict.run(init)
        scale_1, scale_2, scale_3 = sess_predict.run([model.scale_1, model.scale_2, model.scale_3], feed_dict={model.inputs: infer_data})
        anchors_1 = model.anchors[0:3]
        anchors_2 = model.anchors[3:6]
        anchors_3 = model.anchors[6:9]

        box_1, conf_1, class_1, offset_1 = yolo_helper.extract_feature(scale_1, model.num_classes, anchors_1, model_h=model.height, model_w=model.width)
        box_2, conf_2, class_2, offset_2 = yolo_helper.extract_feature(scale_2, model.num_classes, anchors_2, model_h=model.height, model_w=model.width)
        box_3, conf_3, class_3, offset_3 = yolo_helper.extract_feature(scale_3, model.num_classes, anchors_3, model_h=model.height, model_w=model.width)

        # print(scale_1)
        # print(scale_2)
        # print(scale_3)
        # print(anchors_1)
        # print(anchors_2)
        # print(anchors_3)

        print(sess_predict.run(box_1))




    # ToDo: final output decode form extracts


'''--------Test the scale--------'''
if __name__ == "__main__":
    model = YOLOv3(height=1216, width=1920, num_classes=10)
    print(model.scale_1)
    print(model.scale_2)
    print(model.scale_3)

    image_array = np.arange(1 * 1216 * 1920 * 3, dtype=np.float32).reshape((1, 1216, 1920, 3))  # h , w
    infer(model, image_array)

