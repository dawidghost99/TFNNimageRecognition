import AITF as ai
import csv





epoch = 2   

in_neurons = 8

for x in range(3):


    tms, accu= ai.ArtAI(in_neurons,epoch,x)

    with open('results.csv', 'a') as file:
        writer = csv.writer(file)
        #writer.writerow(["model: ", "Input Neurons ", "Epochs", "Time ", "Accuracy"])
        writer.writerow([x, in_neurons, epoch,tms,accu ])
    epoch += 1
    in_neurons += 1






