from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

csv_file = "../data/Admission_Predict.csv"
df = pd.read_csv(csv_file, index_col=0)

df = df.rename(columns={
    "GRE Score":"GRE",
    "TOEFL Score": "TOEFL",
    "University Rating": "Rating",
    "Chance of Admit ": "Chance"
})

def linear_regression_1(x, y):
    """
    Linear regression with 1 feature
    x: feature
    y: response
    """

    # 80% train, 20% test. We not use test set yet
    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

    x_train = x_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)

    lr = LinearRegression().fit(x_train, y_train)
    y_lr_train = lr.predict(x_train)

    print("r_square score: ", r2_score(y_train, y_lr_train))
    print("Coef", lr.coef_)
    print("beta0", lr.intercept_)

    plt.scatter(x_train, y_train, s=40, alpha=1, edgecolors='w', color='r')

    print("Plotting")
    xx = np.linspace(x.min(), x.max(), num=20).reshape(-1, 1)
    yy = lr.predict(xx)
    plt.plot(xx, yy.reshape(-1, 1), color='b')

    plt.xlabel(x.name)
    plt.ylabel(y.name)

    plt.show()

def linear_regression_2(x, y):
    """
    Linear regression with 2 features
    x: pandas DataFrame with 2 columns
    y: 
    """
    # 80% train, 20% test. We not use test set yet
    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

    lr = LinearRegression().fit(x_train, y_train)
    y_lr_train = lr.predict(x_train)

    print("r_square score: ", r2_score(y_train, y_lr_train))
    print("Coef", lr.coef_)
    print("beta0", lr.intercept_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x.name[0], ys=x.name[1], zs=y , s=40, alpha=1, edgecolors='w', color='r')

    print("Plotting")
    xx1, xx2 = np.meshgrid(
        np.linspace(x['GRE'].min(), x['GRE'].max(), num=20),
        np.linspace(x['CGPA'].min(), x['CGPA'].max(), num=20)
    )
    XX = np.column_stack([xx1.ravel(), xx2.ravel()])

    yy = lr.predict(XX)
    ax.plot_surface(xx1, xx2, yy.reshape(xx1.shape), color='b')

    ax.set_xlabel('GRE')
    ax.set_ylabel('CGPA')
    ax.set_zlabel('Chance')

    plt.show()

def linear_regression_all(df: pd.DataFrame):
    """
    Linear regression multiple features

    """
    y = df["Chance"]
    x = df.drop(["Chance"],axis=1)

    # 80% train, 20% test. We not use test set yet
    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

    lr = LinearRegression().fit(x_train,y_train)
    y_lr_train = lr.predict(x_train)

    print("r_square score: ", r2_score(y_train, y_lr_train))
    print("Coef", lr.coef_)
    print("beta0", lr.intercept_)

if __name__ == "__main__":
    print("Linear regression with 1 feature")
    linear_regression_1(df["GRE"], df["Chance"])
    linear_regression_2(df[["GRE", "CGPA"]], df["Chance"])
    linear_regression_all(df)
