"""
1700554


For this project I am building an Artificial Neural Network in python 3.7. I will be using the 
Tensorflow Library, which is an open source 

To see the pyplot images please use https://colab.research.google.com/notebooks/intro.ipynb 


sources:
https://www.tensorflow.org/datasets/catalog/fashion_mnist
https://www.tensorflow.org/datasets/catalog/overview
"""



""" 

Loading all relevant libraries

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


"""loading the dataset """

fashionMinist = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = fashionMinist.load_data()


"""view the training image"""

#imgIndex  = 0
#img = trainImages[imgIndex]
#print("image lable: ", trainLabels[imgIndex])

"""
We only have a total of 10 unique lables between 0-9, 

0 = t-shirt
1 = trouser
2 = sweater
3 = dress
4 = coat
5 = sandal
6 = shirt
7 = trainers
8 = bag
9 = boot


"""

#plt.imshow(img)


""" print the shape of our training images """

print("In the training images dataset you have " + str (trainImages.shape[0] )+ " of " + str (trainImages.shape[1]) + "x" + str(trainImages.shape[2]) + "px images.")
print("In the testing images dataset you have " +  str(testImages.shape[0]) + " of " + str(testImages.shape[1]) + "x" + str(testImages.shape[2]) + "px images.")




"""

create the neural network model 

    keras.layers.Flatten(inputShape=(28,28)), <== this is the input layer. the flatten methond will reduse the Dimensionality the images
    

    keras.layers.Dense(128, activation=tf.nn.relu) <== this is the hidden layer. The 128 refers to how many neurons I have and 'relu' is the activation function

    keras.layers.Dense(10, activation = tf.nn.softmax) <== this is the output layer. this only has 10 neurons as there are only 10 unique lables, 'softmax' is the activation function





"""

model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)



    ])


"""
compiling the model

______________________________________________

***NOTE***


replace 

optimizer = tf.optimizers.Adam(), 

with 

optimizer = tf.train.AdamOptimizer(),


when using goole colab

 _____________________________________________
"""

model.compile(
    
    optimizer = tf.optimizers.Adam(),
    loss  = "sparse_categorical_crossentropy", 
    metrics  = ["accuracy"]


    )


""" Training the model



"""


model.fit(trainImages,trainLabels, epochs=1, batch_size = 32)


"""
evaluating the model
"""

model.evaluate(testImages, testLabels)


"""
making a classification
"""


predicitions = model.predict(testImages[0:9])



print("This shows the category given to the NN: " )
print(testLabels[0:9])


print("This shows the category predicted by the model: " )
print( np.argmax(predicitions, axis=1))


for x in range(0,9):
    Image = testImages[x]
    Image = np.array(Image,dtype="float")
    pixels = Image.reshape((28,28))
    plt.imshow(testImages[x])
    plt.show()
