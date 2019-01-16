from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import samples.utils.common_tool as common

batch_size = 128
num_classes = 10
epochs = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255.
x_test = x_test.reshape(10000, 784) / 255.

# convert class vectors to binary class matrices
y_train = common.onehot_encoding(y_train, num_classes)
y_test = common.onehot_encoding(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

json_string = model.to_json()
open("model/sample.json", 'w').write(json_string)

yaml_string = model.to_yaml()
open("model/sample.yaml", 'w').write(yaml_string)

model.save_weights("model/sample.hdf5")
