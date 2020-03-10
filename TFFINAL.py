import AITF as ai
import csv



"""
epoch variable referes to the amount of learning cycles you want to give the neural network
leanrate referes to the learning rate you'd like the neural network to have
in_neurons refers to the amount of neurons you'd like there to be in the hidden layer

"""

epoch = 10 
learnrate = 0.001 
in_neurons = 64

for x in range(5):

    """
    tms is the time returned from training and testing the neural network
    training_accu is the training accuracy
    test_accu is the test accuracy
    training_loss is the training loss
    testing_loss is the testing loss

    """
       
    tms, training_accu, test_accu, training_loss, testing_loss = ai.ArtAI(in_neurons,epoch,x,learnrate)

    with open('results.csv', 'a') as file:
        writer = csv.writer(file)
        #writer.writerow(["model: ", "Input Neurons ", "Epochs", "Time", "Accuracy"])
        writer.writerow([x, in_neurons, epoch,tms,training_accu,test_accu,training_loss,testing_loss ])
    in_neurons += 64







