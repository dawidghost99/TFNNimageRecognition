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
import random
import time
import os


fashionMinist = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = fashionMinist.load_data()

end = 0.0
start = 0.0



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






def ArtAI(inputNeurons, numepochs,run, LR):

    #starting timer
    start = time.time()


    """
create the neural network model 
    keras.layers.Flatten(inputShape=(28,28)), <== this is the input layer. the flatten methond will reduse the Dimensionality the images
    
    keras.layers.Dense(128, activation=tf.nn.relu) <== this is the hidden layer. The 128 refers to how many neurons I have and 'relu' is the activation function
    keras.layers.Dense(10, activation = tf.nn.softmax) <== this is the output layer. this only has 10 neurons as there are only 10 unique lables, 'softmax' is the activation function
"""






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
 _____________________________________________
"""








    model.compile(
    
    optimizer = tf.optimizers.Nadam(learning_rate=LR),  
    loss  = "sparse_categorical_crossentropy", 
    metrics  = ["accuracy"]
    
    )



    """ Training the model"""




    preditction_history = model.fit(trainImages,trainLabels, 
                            epochs=numepochs, 
                             batch_size = 32,
                             verbose=1, 
                             validation_data=(trainImages, trainLabels))
   

    """
evaluating the model
"""

    model.evaluate(testImages, testLabels)



    """
making a classification
"""
    predicitions = model.predict(testImages[:]) # use all



 
    #ending timer
    end = time.time()



    #training loss
    trainLoss = preditction_history.history['loss']

    #testing loss
    val_loss = preditction_history.history['val_loss']



    trainAccuracy = preditction_history.history['accuracy']



   
    trainAcu = 0.0


    guess = []
    same = 0.00
    guess = np.argmax(predicitions, axis=1) 
    
    count =0


    #training accuracy

    for T in trainAccuracy:
        trainAcu += T

    trainAcu /= len(trainAccuracy)




    # test prediction accuracy

    for y in guess:
       # print(y)
        if testLabels[count] == y:
            same+=1
        count +=1

    testAccu = (same / 10000)*100

 

    # test accuracy



    dirName = 'C:/Users/dawid/Desktop/FINAL AI/AImodels/model' + str(run)
    os.mkdir(dirName)
    model.save(dirName)



    
    print("_________________________________________________________")

    print("In the training images dataset you have " + str (trainImages.shape[0] )+ " of " + str (trainImages.shape[1]) + "x" + str(trainImages.shape[2]) + "px images.")
    print(" ")
    print("In the testing images dataset you have " +  str(testImages.shape[0]) + " of " + str(testImages.shape[1]) + "x" + str(testImages.shape[2]) + "px images.")
    print(" ")
    print("This shows the category given to the NN:        ", testLabels[:] )
    print(" ")
    print("This shows the category predicted by the model: ",  np.argmax(predicitions, axis=1) )
    print("guess", guess )
    tottime =  end- start
    print("time: ", tottime, " Seconds")
    print("same", same , " /10,000")
    print("accuracy: ",testAccu , "%")

    print("_________________________________________________________")

    """ 
    this loop will show the first 10 images the neural network tries to classify after training. The images are taken from the testing tuple.
    """
    for x in range(0,9):
        Image = testImages[x]
        Image = np.array(Image,dtype="float")
        pixels = Image.reshape((28,28))
        plt.imshow(testImages[x])
        plt.show()

    return(tottime, trainAcu,testAccu ,trainLoss,val_loss)










