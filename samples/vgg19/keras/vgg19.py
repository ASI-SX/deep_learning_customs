import keras
from keras import backend as K
from qumico.Qumico import Qumico

# kerasモデル準備
K.set_learning_phase(0)
model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=True, input_tensor=None, input_shape=None)

# qumico インスタンス作成
convert = Qumico()
convert.conv_keras_to_onnx(type="keras", output_path="onnx", model_name="vgg19", cache_path="model", output_op_name=model.output.op.name, K=K)
