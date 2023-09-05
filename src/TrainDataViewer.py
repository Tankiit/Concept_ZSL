# import data file and graph the train/validation accuracy and loss
# columns: epoch, avg_loss, avg_vloss, avg_acc, avg_fp, avg_ma, avg_oa (false positive, missing attributes, output attributes)

import pandas as pd
import matplotlib.pyplot as plt

# import data file
data = pd.read_csv('results/CUBRes18AutoPredData.csv', index_col=None, sep=', ')
data2 = pd.read_csv('results/CUBRes18Cls.csv', index_col=None, sep=', ')

# graph the train/validation accuracy and loss
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('ResNet18 AutoPredicates 96 attributes')
ax1.plot(data['epoch'], data['avg_acc'], label='AP validation')
ax1.plot(data2['epoch'], data2['avg_acc'], label='Cls validation')

ax12 = ax1.twinx()
ax12.plot(data['epoch'], data['avg_fp'], label='false positive', color='red')
ax12.plot(data['epoch'], data['avg_ma'], label='missing attributes', color='green')
ax12.plot(data['epoch'], data['avg_oa'], label='output attributes', color='orange')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')

ax12.set_ylabel('Attributes')
ax1.legend(loc='upper right')
ax12.legend(loc='center right')


# divide ax2 into two plots
ax22 = ax2.twinx()

ax2.plot(data['epoch'], data['avg_loss'], label='AP train', color='blue')
ax2.plot(data['epoch'], data['avg_vloss'], label='AP validation', color='green')
ax22.plot(data2['epoch'], data2['avg_loss'], label='Cls train', color='orange')
ax22.plot(data2['epoch'], data2['avg_vloss'], label='Cls validation', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')


ax2.legend(loc='upper right')
ax22.legend(loc='center right')

plt.savefig('results/CUBRes18LongTrainingData.png')
plt.show()