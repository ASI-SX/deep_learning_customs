import keras
from keras import backend as K

# kerasモデル準備
K.set_learning_phase(0)
model = keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# convert インスタンス作成
# convert = ONNXConvert()
# convert.conv_keras_to_onnx(type="keras", output_path="onnx", model_name="dense_net_201", cache_path="model", output_op_name=model.output.op.name, K=K)
