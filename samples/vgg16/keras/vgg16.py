import keras
from keras import backend as K

# kerasモデル準備
K.set_learning_phase(0)
model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, input_tensor=None, input_shape=None)

# convert インスタンス作成
# convert = ONNXConvert()
# convert.conv_keras_to_onnx(type="keras", output_path="onnx", model_name="vgg16", cache_path="model", output_op_name=model.output.op.name, K=K)
