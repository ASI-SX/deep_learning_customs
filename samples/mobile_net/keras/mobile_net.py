import keras
from keras import backend as K
from qumico.Qumico import Qumico

# kerasモデル準備
K.set_learning_phase(0)
model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

# qumico インスタンス作成
convert = Qumico()
convert.conv_keras_to_onnx(type="keras", output_path="onnx", model_name="mobile_net", cache_path="model", output_op_name=model.output.op.name, K=K)
