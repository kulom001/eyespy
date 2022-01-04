import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras as k
from keras.utils import np_utils
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
plt.style.use('fivethirtyeight')

#load data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(32, (3, 3), padding= 'same', activation= 'relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation= 'relu', padding= 'same'))
model.add(Conv2D(64, (3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation= 'softmax'))

model.compile(
    loss= "categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=32, # varies from 32-120
    epochs=30, #how many time you run through
   validation_data= (x_test, y_test),
    shuffle=True
)

model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

model.save_weights("model_weights.h5")


from keras import models
from keras.models import model_from_json
from pathlib import Path
from keras_preprocessing import image
import numpy as np
class_labels = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)

model.load_weights("model_weights.h5")
img = image.load_img("cat.png", target_size=(32, 32))
image_to_test = image.img_to_array(img) / 255

list_of_images = np.expand_dims(image_to_test, axis=0)

results = model.predict(list_of_images)

single_result = results[0]

most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

class_label = class_labels[most_likely_class_index]
print("this image is of a {}. Likelihood: {:2f}".format(class_label, class_likelihood))