"""
1700554
For this project I am building an Artificial Neural Network in python 3.7. I will be using the Tensorflow Library
This program classifies clothes from the Fashon MNIST data seet
If you do not have the necessarty libraries installed for testing purposes please use https://colab.research.google.com/notebooks/intro.ipynb 
sources:
https://www.tensorflow.org/datasets/catalog/overview
https://www.tensorflow.org/datasets/catalog/fashion_mnist
https://www.tensorflow.org/tutorials/keras/classification
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data?version=nightly
"""


"""
- give user abaility to choose categories
- different optimizers
- different model structures
- model.save() (saves the model and give it data)
- give it a accuracy vs complexity graph



FIND PATTERNS (more nodes and layers) = how that affects accuracy

"""



""" 
Loading all relevant libraries
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import time



"""
loading the dataset 
the dataset is providied by fashon Minst, as of writing this is provideds us with 60,000 training images and 10,000 test images 
The copyright for Fashion-MNIST is held by Zalando SE. Fashion-MNIST is licensed under the MIT license.
(trainImages, trainLabels), (testImages, testLabels) are Tuples
A tuple is a sequence of immutable Python objects. Tuples are sequences, just like lists. The differences between tuples and lists are, the tuples cannot be changed unlike lists and tuples use parentheses
"""

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


"""
create the neural network model 
    keras.layers.Flatten(inputShape=(28,28)), <== this is the input layer. the flatten methond will reduse the dimensionality the images
    
    keras.layers.Dense(128, activation=tf.nn.relu) <== this is the hidden layer. The 128 refers to how many neurons I have and 'relu' is the activation function. Relu meaning rectified linear unit

    keras.layers.Dense(10, activation = tf.nn.softmax) <== this is the output layer. this only has 10 neurons as there are only 10 unique lables, 'softmax' is the activation function
"""

def ArtAI(inputNeurons, numepochs):


    start = time.time()

    model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(inputNeurons, activation=tf.nn.relu),
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
    ______________________________________________
    optimizers available are 
    -Stochastic Gradient descent
    -Stochastic Gradient descent with gradient clipping
    -Momentum
    -Nesterov momentum
    -Adagrad
    -Adadelta
    -RMSProp
    -Adam
    -Adamax
    -SMORMS3
    for this I am using Adam as that is what TensorFlow recommend best for object detection
    Optimizer seeks to minimize the loss function. It is also how the model is updated based on the data it sees and its loss function.
    Loss function measures how accurate the model is during training. This "steers" the model into the right direction. 
    Metrics is used to monitor the training and testing steps. For this I am using accuracy, the fraction of the images that are correctly classified.
    """
    model.compile(
    
    optimizer = tf.optimizers.Adam(),
    loss  = "sparse_categorical_crossentropy", 
    metrics  = ["accuracy"]
    
    )


    """ 
    Training the model
    epochs are training cycles
    if epochs=5 then the AI has 5 training cycles to learn the training images.
    The tells us the number of samples per gradient of training. 32 is the default and can be left out
    """


    model.fit(trainImages,trainLabels, 
          epochs=numepochs, 
          batch_size = 32)


    """
    evaluating the model
    we are evaluatuing the model on the test data set
    """

    model.evaluate(testImages, testLabels)

#print("Evaluation: ",model.evaluate(testImages, testLabels,) )

    """
    making a predition
    classifying the images it sees.
    """

    randominterget = random.randint(0,9992)

    randomintergetend = randominterget + 9

    predicitions = model.predict(testImages[randominterget:randomintergetend])



    end = time.time()
    print( "end - start",  end - start)


    print("_________________________________________________________")

    print("In the training images dataset you have " + str (trainImages.shape[0] )+ " of " + str (trainImages.shape[1]) + "x" + str(trainImages.shape[2]) + "px images.")
    print(" ")
    print("In the testing images dataset you have " +  str(testImages.shape[0]) + " of " + str(testImages.shape[1]) + "x" + str(testImages.shape[2]) + "px images.")
    print(" ")
    print("random starting postition of the testImages Tuple was ", randominterget)
    print(" ")
    print("random ending postition of the testImages Tuple was   ", randomintergetend)
    print(" ")
    print("This shows the category given to the NN:        ", testLabels[randominterget:randomintergetend] )
    print(" ")
    print("This shows the category predicted by the model: ",  np.argmax(predicitions, axis=1) )

    print("_________________________________________________________")



    """for x in range(0,9):
            Image = testImages[x]
            Image = np.array(Image,dtype="float")
            pixels = Image.reshape((28,28))
            plt.imshow(testImages[x])
            plt.show()
    




"""






""" ___________________________________________________________________ """




ArtAI(128,3)



x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)

plt.show()
