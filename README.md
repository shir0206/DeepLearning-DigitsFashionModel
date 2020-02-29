#  Deep Learning - Digits & Fashion Model

In this project I used Deep Learning techniques in order to recognize images. The dataset of the images were taken from `MNIST` dataset.
After training the model, I tried to predict the classification of the test images, and then calculated the result of the model accuracy.</br>
The detailed results is available in `Project Sheet.pdf	`.
</br></br>

## :bar_chart: (1) Training & Testing Fashion Images
The model was trained using 10 epochs and reached an accuracy of 91.08% on the training data.
The accuracy of the test dataset reached 88%.

![Ex02_Q01_Output05](https://user-images.githubusercontent.com/40990488/75606045-92848d00-5af1-11ea-8076-bf6172cdb4b5.png)


## :bar_chart: (2) Training & Testing Hand-Written Digit Images
The model was trained using 10 epochs and reached an accuracy of 99.56% on the training data.
The accuracy of the test dataset reached 97.87%.

![Ex02_Q02_Output05](https://user-images.githubusercontent.com/40990488/75606059-adef9800-5af1-11ea-9cd8-c5fba4f2cc73.png)

## :bar_chart: (3) Improve The Hand-Written Digit Images Model
I changed the network by enlarging the training: instead of 10 epochs - I used 200 epochs. </br>
In addition, instead of 128 layers, I used 300 layers.</br>
The model was reached an accuracy of 100%  on the training data (In comparison to the accuracy of 99.56% reached in previous model). </br>
The accuracy of the test dataset reached 98.41% (In comparison to the accuracy of 97.87% reached in previous model).</br>

## :bar_chart: (4) Accuracies Histogram
Using the previous improvement, I plotted an histogram that represents the model accuracy.</br>
According to the histogram of the accuracies, 98.41% of the test images were correct and 1.59% were incorrect.
![Ex02_Q04_Output04](https://user-images.githubusercontent.com/40990488/75606155-a381ce00-5af2-11ea-9a4f-5efffc62b060.png)

## :bar_chart: (5) Training & Testing Different Types Of Datasets
Since the model was trained on one dataset and tested on another dataset - the model does not really know how to distinguish the different classifications of each image in the test dataset.</br>
The accuracy obtained seems to be random. Which means, the results the model was "correct" were random. </br>
In another run of the model, I may get different accuracy.
<br>
### Part A - Training Hand-Written Digit Images & Testing Fashion Images
![Ex02_Q05_A_Output05](https://user-images.githubusercontent.com/40990488/75606342-569ef700-5af4-11ea-837a-052326f8e30c.png)

### Part B - Training Fashion Images & Testing Hand-Written Digit Images
![Ex02_Q05_B_Output05](https://user-images.githubusercontent.com/40990488/75606361-7a623d00-5af4-11ea-8a7b-3c3f89d4cba1.png)

## :bar_chart: (6) Gereation New Images Dataset Using 2 Images & Alpha (0<Alpha<1)
I trained the model on the dataset of images of digits.
Then, I generated a new test dataset of 100 images, where each image was generated from a weighted average of 2 images in the training dataset: </br>
`I(α)= α⋅I1  +(1-α)⋅I2` `α∈[0,1)` .</br>
The first image was taken from the test dataset and was an image of the digit ‘2’.</br>
The second image was taken from the test dataset and was an image of the digit ‘1’.</br>
the accuracy of the test dataset reached 71%.
![Ex02_Q06_Output08](https://user-images.githubusercontent.com/40990488/75606447-6cf98280-5af5-11ea-906e-e9c47fee80bb.png)



## :bar_chart: (7) Extract The Results Of The 2nd Layer
Using the previous model, I ran the algorithm once again and printed the images via representation in an array.
Then, I extracted the results of the second layer for each of them. 


## :bar_chart: (8) Predict Results Using The Last Layer
Using the previous model, I ran the algorithm once again and  extracted the results of the last layer for each of the test images.</br>
In each image, the last layer contains an array of probabilities - where each index represents the class possibilities of what the image can be, and each value in the array represents the probability that what is shown in the image is that class.</br>
Then, I compared the classified results of the last layer to the true results of the test images.</br>
The calculated accuracy and the accuracy of the model were equal.


## :bar_chart: (9) Gereation New Images Dataset Using 2 Images & Alpha (-1<Alpha<2)
I trained the model on the dataset of images of digits.
Then, I generated a new test dataset of 100 images, where each image was generated from a weighted average of 2 images in the training dataset: </br>
`V(α)= α⋅V1  +(1-α)⋅V2` `α∈[-1,2)` .</br>
the accuracy of the test dataset reached 66%.
![Ex02_Q09_Output08](https://user-images.githubusercontent.com/40990488/75606530-5acc1400-5af6-11ea-919e-42e7ac870184.png)


##

#### :pushpin: Source
https://www.tensorflow.org/tutorials/keras/classification
