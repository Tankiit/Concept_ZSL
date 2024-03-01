import pandas as pd

df = pd.read_csv('/home/kyucilow/Code/AI/BSS/Concept_ZSL/results/CZSL - Results - ZSL.csv', header=None)

# only keep rows where 9th column is using 2*loss_mean_attr
dataset = 'CUB'
df = df[df[9] == 'using 2*loss_mean_attr']
df = df[df[0] == dataset]

# only keep columns where 6th column == 7th column
df[5] = df[5].str.replace('features', '').astype(int)
df = df[df[6] == df[7]]
df = df[df[5] < 300]

# 5th column is the feature count. Calculate mean and std of 4th column (accuracy) for each feature count
df = df[[2, 3, 4, 5]]
df[2] = df[2].str.replace('%', '').astype(float)
df[3] = df[3].str.replace('%', '').astype(float)
df[4] = df[4].str.replace('%', '').astype(float)
df = df.groupby(5).agg(['mean', 'std'])

print(df)

import matplotlib.pyplot as plt

plt.errorbar(df.index, df[2]['mean'], yerr=df[2]['std'], fmt='o', label='Seen')
plt.errorbar(df.index, df[3]['mean'], yerr=df[3]['std'], fmt='o', label='Unseen')
plt.errorbar(df.index, df[4]['mean'], yerr=df[4]['std'], fmt='o', label='Harmonic Mean')

plt.legend()

plt.xticks(df.index)

plt.xlabel('Feature Count')
plt.ylabel('Accuracy')
plt.title(f'Accuracy for different Feature Counts ({dataset})')

plt.show()