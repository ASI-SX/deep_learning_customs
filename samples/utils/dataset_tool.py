import numpy as np
import random


class DatasetTool():
    def __init__(self, training_flag=False, data=None, label=None, repeat=False, one_hot_classes=None):
        super(DatasetTool, self).__init__()
        self.training = training_flag
        self.data = data
        self.label = label
        self.one_hot_classes = one_hot_classes
        self.repeat = repeat
        self.total_size = data.shape[0]
        self.index_list = list(np.arange(0, self.total_size))
        np.random.shuffle(self.index_list)

        if training_flag:
            try:
                assert data.shape[0] == label.shape[0], "学習データサイズが異なる, 入力データサイズ = {0} ラベルサイズ = {1}".format(data.shape[0], label.shape[0])
            except AssertionError as err:
                print("Error : ", err)

    def next_batch(self, batch_size):
        try:
            assert batch_size < self.total_size, "batch_sizeがtotal_batchに超えている。"
        except AssertionError as err:
            print("Error : ", err)
        if self.repeat:
            batch_mask = random.choices(self.index_list, k=batch_size)
        else:
            batch_mask = random.sample(self.index_list, k=batch_size)

        if self.training:
            x_batch = self.data[batch_mask]
            y_batch = self.label[batch_mask]
            if self.one_hot_classes is not None:
                y_batch = np.identity(self.one_hot_classes)[y_batch]
            return x_batch, y_batch
        else:
            x_batch = self.data[batch_mask]
            return x_batch

    def next_batch_once(self, batch_size):
        if len(self.index_list) == 0:
            self.index_reset()
        batch_mask = self.index_list[:batch_size]
        self.index_list = self.index_list[batch_size:]

        if self.training:
            x_batch = self.data[batch_mask]
            y_batch = self.label[batch_mask]
            if self.one_hot_classes is not None:
                y_batch = np.identity(self.one_hot_classes)[y_batch]
            return x_batch, y_batch
        else:
            x_batch = self.data[batch_mask]
            return x_batch

    def index_reset(self):
        self.index_list = list(np.arange(0, self.total_size))
