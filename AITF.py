
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

def ArtAI(inputNeurons, numepochs,run):


    start = time.time()

    model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(inputNeurons, activation=tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)

    ])

    model.compile(
    
    optimizer = tf.optimizers.Adamax(),
    loss  = "sparse_categorical_crossentropy", 
    metrics  = ["accuracy"]
    
    )



    preditction_history = model.fit(trainImages,trainLabels, 
                            epochs=numepochs, 
                             batch_size = 32,
                             verbose=1, 
                             validation_data=(trainImages, trainLabels))
   


    model.evaluate(testImages, testLabels)

    predicitions = model.predict(testImages[:]) # use all



 
    
    end = time.time()




    loss = preditction_history.history['loss']
    val_loss = preditction_history.history['val_loss']

    trainAccuracy = preditction_history.history['accuracy']



    trainLoss = 0.0
    testLoss = 0.0
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


    #average train loss

    for L in loss:
        trainLoss += L

    trainLoss /= len(loss)



    # average test loss

    for L in val_loss:
        testLoss += L

    testLoss /= len(val_loss)

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








    #       tms, training_accu, test_accu, training_loss,testing_loss 


    return(tottime, trainAcu,testAccu ,trainLoss,testLoss)








"""
times = end - start
#x = np.arange(0, 5, times)
y = np.sin(x)
plt.plot(x, y)
plt.show()
"""

