import keras
from keras import backend as K
from qumico.Qumico import Qumico

# kerasモデル準備
K.set_learning_phase(0)
model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# qumico インスタンス作成
convert = Qumico()
convert.conv_keras_to_onnx(type="keras", output_path="onnx", model_name="res_net_50", cache_path="model", output_op_name=model.output.op.name, K=K)
