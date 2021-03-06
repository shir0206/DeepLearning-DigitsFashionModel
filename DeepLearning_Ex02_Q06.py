
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.stdout = open('Q6.txt', 'w')

print(tf.__version__)

#Import the Fashion MNIST dataset¶

digit_minst = keras.datasets.mnist

# The train_images and train_labels arrays are the training set—the data the model uses to learn.
# The model is tested against the test set, the test_images, and test_labels arrays.


(train_images, train_labels), (test_images, test_labels) = digit_minst.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



# Create set of test images: I(alpha) = alpha * I1 + (1-alpha) * I2

img1 = test_images[1]
label1 = test_labels[1]

img2 = test_images[2]
label2 = test_labels[2]

print('img1 = ', img1)
print('label1 = ', label1)

plt.figure()
plt.imshow(img1)
plt.colorbar()
plt.grid(False)
plt.show()

print()

print('img2 = ', img2)
print('label2 = ', label2)

plt.figure()
plt.imshow(img2)
plt.colorbar()
plt.grid(False)
plt.show()

# Create new array
test_images_with_alpha_temp=[]
test_labels_with_alpha_temp=[]

test_images_with_alpha = np.array([])
test_labels_with_alpha = np.array([])

# Alpha is a number between 0 and 1
for number in range(0, 100, 1):

    # Create alpha
    alpha = (number/100)

    # Create new image
    img_new = ((img1 * alpha) + (img2 * (1 - alpha))) / 2

    print(img_new)

    if (number < 50):
        label_new = label2
    else:
        label_new = label1

    # Add new image to the list
    test_images_with_alpha_temp.append(img_new)

    # Add new image label to the array
    test_labels_with_alpha_temp.append(label_new)
    #test_labels_with_alpha = np.append(test_labels_with_alpha, label_new)

print('test_images_with_alpha_temp', test_images_with_alpha_temp)

# Create array of the generated images
test_images_with_alpha = np.stack(test_images_with_alpha_temp, axis=0)
print('test_images_with_alpha', test_images_with_alpha)

test_labels_with_alpha = np.stack(test_labels_with_alpha_temp, axis=0)
print('test_labels_with_alpha', test_labels_with_alpha)

# Explore the data¶

train_images.shape

len(train_labels)

train_labels

test_images_with_alpha.shape

len(test_labels_with_alpha)


#Preprocess the data¶

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


train_images = train_images / 255.0

test_images_with_alpha = test_images_with_alpha / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Build the model¶

# Set up the layers¶

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model¶

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Train the model¶

model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy¶
print('Test images', test_images_with_alpha)
print('Test labels', test_labels_with_alpha)

test_loss, test_acc = model.evaluate(test_images_with_alpha,  test_labels_with_alpha, verbose=2)

p = model.predict(test_images_with_alpha)
p = np.argmax(p,axis=1)
test_acc1 = 1-np.count_nonzero(p-test_labels_with_alpha)/len(test_labels_with_alpha)

print('\nTest accuracy (original):', test_acc, '\nTest accuracy (manual):', test_acc1)


# Plot accuracy¶
x = np.arange(2)
plt.bar(x, height= [test_acc , 1-test_acc])
plt.xticks(x, ['Correct', 'Incorrect'])

# Make predictions¶
predictions = model.predict(test_images_with_alpha)
predictions[0]
np.argmax(predictions[0])
test_labels_with_alpha[0]

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

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
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
plot_image(i, predictions[i], test_labels_with_alpha, test_images_with_alpha)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels_with_alpha)
plt.show()



i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels_with_alpha, test_images_with_alpha)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels_with_alpha)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels_with_alpha, test_images_with_alpha)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels_with_alpha)
plt.tight_layout()
plt.show()




# Grab an image from the test dataset.
img = test_images_with_alpha[1]

print(img.shape)




# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


predictions_single = model.predict(img)

print(predictions_single)



plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)



np.argmax(predictions_single[0])

