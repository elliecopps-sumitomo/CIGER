import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_1 = open('./output/trained_100_epochs.txt', 'r')
training_2 = open('./output/best_with_lr003.txt', 'r')

training_1 = training_1.read()
training_2 = training_2.read()

training_1 = training_1.splitlines()
training_2 = training_2.splitlines()

training_loss = []
ndcg = []

for i in range(9, 1496, 15):
    training_loss.append(float(training_1[i]))
    ndcg.append(float(training_1[i + 8].split(':')[1]))

for i in range(9, 296, 15):
    training_loss.append(float(training_2[i]))
    ndcg.append(float(training_2[i + 8].split(':')[1]))

fig, ax = plt.subplots()
lossPlt = ax.plot(training_loss, color='r', label="Training Loss")
ndcgPlt = ax.plot(ndcg, color='b', label="NDCG")
#ax.set_ylabel("Training Loss")
ax.set_xlabel("epoch")
ax.legend(['Training Loss', 'NDCG'])

fig.savefig("training_metrics.png")