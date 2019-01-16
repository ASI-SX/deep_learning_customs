import keras
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

# kerasモデル準備
K.set_learning_phase(0)
model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, input_tensor=None, input_shape=None)


img_path = 'data_test/001.JPEG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# features = model.predict(x)
# result = decode_predictions(features, top=5)
# print(result)