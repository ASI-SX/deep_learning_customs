import tensorflow as tf
import numpy as np
from samples.yolo2.tensorflow.tiny_yolo2_model import TINY_YOLO_v2
import samples.utils.pre_process_tool as list_reader
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool
# from qumico.Qumico import Qumico


def train(model, train_data, epochs, batch_size, save_flag=True):
    tf.reset_default_graph()
    ckpt_file = None
    pb_file = None
    data_size = train_data.total_size
    total_batch = data_size // batch_size

    with tf.Session(graph=model.graph) as sess_train:
        sess_train.run(model.init)
        for epoch in range(epochs):
            total_loss = 0
            total_class_loss = 0
            total_obj_loss = 0
            total_coord_loss = 0
            for i in range(total_batch):
                batch_x, batch_y, _, _ = train_data.next_batch_once(batch_size=batch_size)
                y_true, y_mask = model.pre_process_true_boxes(batch_y, model.grid_w, model.grid_h)

                _, output, loss, class_loss, obj_loss, coord_loss = sess_train.run(
                    [model.train_op, model.output, model.total_loss, model.class_loss, model.object_loss, model.coord_loss],
                    feed_dict={model.inputs: batch_x, model.y_true: y_true, model.y_mask: y_mask})

                total_loss += loss
                total_class_loss += class_loss
                total_obj_loss += obj_loss
                total_coord_loss += coord_loss

                print("=" * 3, "epoch:", epoch + 1, " - batch", i + 1, "/", total_batch, "=" * 3)

            print("total loss : ", total_loss / data_size, " class loss : ", total_class_loss / data_size,
                  " obj loss : ", total_obj_loss / data_size, " coord_loss : ", total_coord_loss / data_size)

        if save_flag:
            ckpt_file = "model/tiny_yolo2.ckpt"
            pb_path = "model"
            pb_name = "tiny_yolo2.pb"
            pb_file = pb_path + "/" + pb_name
            model.saver.save(sess_train, ckpt_file)
            model.pb_saver.write_graph(sess_train.graph, pb_path, pb_name, as_text=False)

    return ckpt_file, pb_file


if __name__ == '__main__':


    voc2007_classes = ['chair', 'bird', 'sofa', 'bicycle', 'cat', 'motorbike', 'bus', 'boat', 'sheep', 'bottle', 'cow',
                       'person', 'horse', 'diningtable', 'pottedplant', 'aeroplane', 'car', 'train', 'dog', 'tvmonitor']

    num_classes = len(voc2007_classes)

    root_path = "train_data_mini/"
    data_list_path = root_path + "images"
    label_list_path = root_path + "annotations"

    data_list = np.asarray(list_reader.get_data_path_list(data_list_path)[:50])

    label_list = np.asarray(list_reader.get_data_path_list(label_list_path)[:50])

    annotation_dataset_tool = AnnotationDatasetTool(training_flag=True, data_list=data_list, label_list=label_list,
                                                    category_class=voc2007_classes, one_hot_classes=num_classes,
                                                    resize_flag=True, target_h=416, target_w=416,
                                                    label_file_type="voc_xml", format="NCHW")

    batch_size = 1
    epoch_num = 100
    total_size = annotation_dataset_tool.total_size

    range_size = int(total_size / batch_size)
    # load model
    tiny_yolo_2 = TINY_YOLO_v2(height=416, width=416, output_op_name="output", num_classes=20, is_train=True,
                               batch_size=batch_size)
    # train and save ckpt pb file
    ckpt_file, pb_file = train(tiny_yolo_2, annotation_dataset_tool, epoch_num, batch_size, save_flag=True)

    print("ckpt_file", ckpt_file)
    print("pb_file", pb_file)

    # prepare Qumico Convertor
    # converter = Qumico()
    # converter.conv_tf_to_onnx(output_path="onnx", model_name="tiny_yolo_2", output_op_name="output",
    #                           cache_path="model", ckpt_file=ckpt_file, pb_file=pb_file)
