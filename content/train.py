import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
X_train=[]
X_test=[]
for i,x in enumerate(x_train):
	X_train+=[cv2.resize(x_train[i],(224,224),interpolation=cv2.INTER_AREA ) ]
for i,x in enumerate(x_test):
	X_test+=[cv2.resize(x_test[i],(224,224),interpolation=cv2.INTER_AREA )]

X_train=np.array(X_train)/255
X_train=X_train.reshape( (-1,224,224,1) )
X_test=np.array(X_test)/255
X_test=X_test.reshape( (-1,224,224,1) )

X_val=X_test[:5000]
y_val=y_test[:5000]
print(X_train.shape)
plt.imshow(X_train[0])
plt.show()
model = Sequential([
  layers.Input( shape=(224, 224,1)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(10,activation='softmax')
])


model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, batch_size=64, epochs=4, validation_data=(X_val, y_val))

fig, axs = plt.subplots(2, 1, figsize=(15,15))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train', 'Val'])

axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])

model.evaluate(X_test, y_test)
model.save('model.h5')
