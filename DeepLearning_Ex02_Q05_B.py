
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pylab as pl

print(tf.__version__)

#Import the Fashion MNIST dataset

digit_minst = keras.datasets.mnist
fashion_mnist = keras.datasets.fashion_mnist

# The train_images and train_labels arrays are the training set—the data the model uses to learn.
# The model is tested against the test set, the test_images, and test_labels arrays.

(digit_train_images, digit_train_labels), (digit_test_images, digit_test_labels) = digit_minst.load_data()
(fashion_train_images, fashion_train_labels), (fashion_test_images, fashion_test_labels) = fashion_mnist.load_data()

digit_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Explore the data

fashion_train_images.shape

len(fashion_train_labels)

fashion_train_labels

fashion_test_images.shape

len(fashion_test_labels)


#Preprocess the data

plt.figure()
plt.imshow(fashion_train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

fashion_train_images = fashion_train_images / 255.0

fashion_test_images = fashion_test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(fashion_train_images[i], cmap=plt.cm.binary)
    plt.xlabel(fashion_class_names[fashion_train_labels[i]])
plt.show()


# Build the model

# Set up the layers

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model

model.fit(fashion_train_images, fashion_train_labels, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(fashion_test_images,  fashion_test_labels, verbose=2)

p = model.predict(fashion_test_images)
p = np.argmax(p,axis=10)
test_acc1 = 1-np.count_nonzero(p-fashion_test_labels)/len(fashion_test_labels)


print('\nTest accuracy (original):', test_acc, '\nTest accuracy (manual):', test_acc1)

# Plot accuracy

x = np.arange(2)
plt.bar(x, height= [test_acc , 1-test_acc])
plt.xticks(x, ['Correct', 'Incorrect'])



# Make predictions
predictions = model.predict(digit_test_images)
predictions[0]
np.argmax(predictions[0])
digit_test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(digit_class_names[predicted_label],
                                100*np.max(predictions_array),
                                digit_class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], digit_test_labels, digit_test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  digit_test_labels)
plt.show()



i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], digit_test_labels, digit_test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  digit_test_labels)
plt.show()



# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], digit_test_labels, digit_test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], digit_test_labels)
plt.tight_layout()
plt.show()




# Grab an image from the test dataset.
img = digit_test_images[1]

print(img.shape)




# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


predictions_single = model.predict(img)

print(predictions_single)



plot_value_array(1, predictions_single[0], digit_test_labels)
_ = plt.xticks(range(10), digit_class_names, rotation=45)



np.argmax(predictions_single[0])


# Plot accuracy histogram

#digit_minst.drop('digit_test_labels' ,axis=1).hist(bins=30, figsize=(9,9))
#pl.suptitle("Histogram for each numeric input variable")
#plt.savefig('digit_hist')
#plt.show()
