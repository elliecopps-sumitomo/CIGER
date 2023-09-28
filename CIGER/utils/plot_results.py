import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

predictions = np.load('../output/predictions/train_predictions.npy')
real_vals = np.load('../output/predictions/train_labels_real.npy')
print(np.shape(predictions))
corr_total = 0
for i in range(0, 610):
    prediction = predictions[i]
    real = real_vals[i]

    # fig, ax = plt.subplots()
    # plot = ax.scatter(prediction, real, s=10)
    r, p = stats.pearsonr(prediction, real)
    corr_total += r
    # plt.annotate('r = {:.2f}'.format(r), xy=(0.7, 0.9), xycoords='axes fraction')

    # fig.savefig('drug_' + str(i) + '.png')
print("Average correlation: ", corr_total/610)