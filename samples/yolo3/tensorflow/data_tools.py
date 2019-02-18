import numpy as np
import samples.utils.pre_process_tool as list_reader
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool
import chainer
from chainercv.links.model.yolo import YOLOv3
from chainercv.links.model.ssd import multibox_loss
import samples.utils.box_convert as bc
from copy import deepcopy

class MultiboxTrainChain(chainer.Chain):
    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


root_path = "/home/deven/traindata/"
data_list_path = root_path + "DTC/dtc_train_images"
label_list_path = root_path + "DTC/dtc_train_annotations"

data_list = np.asarray(list_reader.get_data_path_list(data_list_path))
label_list = np.asarray(list_reader.get_data_path_list(label_list_path))

model = YOLOv3(n_fg_class=10)
model.nms_thresh = 0.5
model.score_thresh = 0.5

annotation_dataset_tool = AnnotationDatasetTool(training_flag=True, data_list=data_list, label_list=label_list,
                                                one_hot_classes=10, resize_flag=True, target_h=1216, target_w=1920)
batch_size = 100
total_size = annotation_dataset_tool.total_size
range_size = total_size / batch_size

print(total_size)
print(batch_size)
print(range_size)

for i in range(range_size):
    train_x, train_y = annotation_dataset_tool.next_batch(batch_size=batch_size)
    print(train_y)
    # for item in train_y:
    #     print("============origin bbox===============")
    #     item_temp = deepcopy(item)
    #     classes = item[..., 0:10]
    #     bbox = item[..., -4:]
    #     print(item_temp)
    #     print("============convert to anbox===============")
    #
    #     anbox = bc.bbox_to_anbox(item_temp)
    #     print(anbox)
    #
    #     print("============convert back to bbox===============")
    #     recall_bbox = bc.anbox_to_bbox(anbox)
    #     print(recall_bbox)


#
# train_chain = MultiboxTrainChain(model)
#
# gpu = 1
# batchsize = 10
# num_epochs = 100
#
# if gpu:
#     gpu_id = 0
#     model.to_gpu()
# else:
#     gpu_id = -1
# optimizer = chainer.optimizers.MomentumSGD(lr=0.0006)
# optimizer.setup(train_chain)
#
# for param in train_chain.params():
#     if param.name == 'b':
#         param.update_rule.add_hook(GradientScaling(2))
#     else:
#         param.update_rule.add_hook(WeightDecay(0.0001))
#
# updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
# trainer = training.Trainer(updater, (num_epochs, 'epoch'), 'results')
#
# log_interval = 1, 'epoch'
# trainer.extend(
#     DetectionVOCEvaluator(val_iter, model, use_07_metric=False, label_names=annotation_dataset_tool.category_class),
#     trigger=log_interval)
# trainer.extend(extensions.LogReport(trigger=log_interval))
# trainer.extend(extensions.observe_lr(), trigger=log_interval)
# trainer.extend(extensions.PrintReport(
#     ['epoch', 'iteration', 'lr', 'main/loss', 'main/loss/loc', 'main/loss/conf', 'validation/main/map']),
#                trigger=log_interval)
# trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'), trigger=(5, 'epoch'))
#
