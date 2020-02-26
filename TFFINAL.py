import AITF as ai
import csv





epoch = 40  

in_neurons = 64

for x in range(5):


    tms, training_accu, test_accu, training_loss, testing_loss = ai.ArtAI(in_neurons,epoch,x)

    with open('results.csv', 'a') as file:
        writer = csv.writer(file)
        #writer.writerow(["model: ", "Input Neurons ", "Epochs", "Time", "Accuracy"])
        writer.writerow([x, in_neurons, epoch,tms,training_accu,test_accu,training_loss,testing_loss ])
    #epoch += 10
    in_neurons += 64







