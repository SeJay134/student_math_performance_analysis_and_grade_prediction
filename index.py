import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

os.makedirs('outputs', exist_ok=True)

try:
    df = pd.read_csv('student_performance_math.csv', sep=';')
except Exception as e:
    print(f'error {e}')

print('df.info')
df.info()

print('df.shape\n', df.shape)
print('df.head(5)\n', df.head(5))

print('df.dtypes\n', df.dtypes)
print('df.isna().sum()\n', df.isna().sum())
print('df.duplicated().sum()\n', df.duplicated().sum())

counts, bins = np.histogram(df['G3'], bins=[x-0.5 for x in range(0,22)])

plt.figure(figsize=(7, 5))
plt.hist(df['G3'], bins=[x - 0.5 for x in range(0, 22)], edgecolor='black', linewidth=0.5)

for count, bin_start in zip(counts, bins[:-1]):
    plt.text(
        x=bin_start + 0.5,
        y=count + 0.2,
        s=int(count),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title('Distribution of Final Math Grades')
plt.xticks(range(0, 21, 1))
plt.xlabel('Final Grade (G3)')
plt.ylabel('Number of Students')
plt.savefig('outputs/g3_distribution.png')
plt.show()

