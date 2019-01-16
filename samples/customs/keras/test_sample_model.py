from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.models import model_from_json
import samples.utils.common_tool as common

# qu classes
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784) / 255.

# convert class vectors to binary class matrices
y_test = common.onehot_encoding(y_test, num_classes)

json_file = "model/sample.json"
h5_file = "model/sample.hdf5"

model = model_from_json(open(json_file).read())
model.load_weights(h5_file)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

(loss, score) = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy : ", score)
