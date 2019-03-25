import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "Admission_Predict.csv"
df = pd.read_csv(csv_file, index_col=0)

df = df.rename(columns={
    "GRE Score":"GRE",
    "TOEFL Score": "TOEFL",
    "University Rating": "Rating",
    "Chance of Admit": "Chance"
    })

sns.pairplot(df)

plt.show()