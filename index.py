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

print('df.shape', df.shape)
df_G3_filtered = df[df['G3'] != 0].copy()
print('df_G3_filtered', df_G3_filtered.shape)

df_G3_filtered['sex'] = df_G3_filtered['sex'].replace({'F': 0, 'M': 1})
cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in cols:
    df_G3_filtered[col] = df_G3_filtered[col].replace({'yes': 1, 'no': 0})

print('df_G3_filtered.info')
df_G3_filtered.info()

print('df_G3_filtered.shape\n', df_G3_filtered.shape)
print('df_G3_filtered.head(5)\n', df_G3_filtered.head(5))
print('df_G3_filtered.dtypes\n', df_G3_filtered.dtypes)

df_corr_not_filtered = df['absences'].corr(df['G3'])
print('df_corr_not_filtered', df_corr_not_filtered)
df_corr_filtered = df_G3_filtered['absences'].corr(df_G3_filtered['G3'])
print('df_corr_filtered', df_corr_filtered)

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].scatter(df['absences'], df['G3'], alpha=0.5)
axs[0].set_title('Original G3 and absences are not filtered')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].scatter(df_G3_filtered['absences'], df_G3_filtered['G3'], alpha=0.5)
axs[1].set_title('G3 and absences are filtered')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.show()

# markdown
# Students with G3=0 likely didn't take the exam.
# Many of them have varying numbers of absences, which adds noise.
# This weakens the correlation between absences and G3 in the original dataset.
# After removing them, the relationship becomes clearer.

columns = ['sex', 'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'schoolsup', 'internet', 'higher', 'activities', 'absences', 'freetime', 'goout', 'Walc', 'G1', 'G2']
corr_columns = []

for col in columns:
    df_corr_col = df_G3_filtered[col].corr(df_G3_filtered['G3'])
    corr_columns.append(f'{col}: {df_corr_col}')
    #print(f"{col}: {df_corr_col}")

corr_columns_sorted = sorted(corr_columns, key=lambda x: float(x.split(': ')[1]))
for value in corr_columns_sorted:
    print(value)

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].scatter(df_G3_filtered['failures'], df_G3_filtered['G3'], alpha=0.5)
axs[0].set_title('G3 and failures')
axs[0].set_xlabel('Failures')
axs[0].set_ylabel('Final Grade (G3)')

axs[1].scatter(df_G3_filtered['schoolsup'], df_G3_filtered['G3'], alpha=0.5)
axs[1].set_title('G3 and schoolsup')
axs[1].set_xlabel('School Support (0/1)')
axs[1].set_ylabel('Final Grade (G3)')
plt.tight_layout()

plt.savefig('outputs/g3_Failures_School_Support.png')
plt.show()

# markdown
# The strongest predictors of G3 are G2 and G1 (grades from previous assessments).
# Negative correlations for failures, schoolsup, and absences indicate that 
# more failures or absences → lower final grade.
# Small positive correlations (sex, internet, higher) suggest weak influence.

# A negative linear relationship is visible: more failures correspond to lower final grades.
# Most students with zero or few failures have high grades.

# Most important predictors of G3 are previous grades (G1, G2), failures, schoolsup, and absences.
# Filtering out G3=0 is important because students who didn’t take the final exam distort 
# correlations.
# Least influential features include freetime, activities, and sex.

X_q4 = df_G3_filtered[['failures']]
y_q4 = df_G3_filtered['G3']

X_train_q4, X_test_q4, y_train_q4, y_test_q4 = train_test_split(
    X_q4, y_q4, test_size=0.2, random_state=42
)

model_q4 = LinearRegression()
model_q4.fit(X_train_q4, y_train_q4)
y_pred_q4 = model_q4.predict(X_test_q4)

print('Slope_q4:', model_q4.coef_[0])
print('Intercept_q4:', model_q4.intercept_, '\n')

rmse_q4 = np.sqrt(np.mean((y_pred_q4 - y_test_q4) ** 2))
print(f'rmse_q4: {rmse_q4:.4f}')
score_q4 = model_q4.score(X_test_q4, y_test_q4)
print(f'score_q4: {score_q4:.4f} \n')

# markdown
# The slope is negative, meaning that each additional failure reduces 
# the final grade by about 1.4 points.

# Since grades are on a 0–20 scale, this is a noticeable but not huge effect.
# The RMSE shows the average prediction error.

# For example, if RMSE ≈ 3, it means predictions are off by about ±3 grade points, 
# which is quite large relative to the scale.

# The R² score is relatively low, meaning that failures alone explains only 
# a small portion of the variation in final grades.

# This is expected based on exploratory analysis, since stronger predictors 
# like G1 and G2 showed much higher correlations with G3.