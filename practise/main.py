from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

csv_file = "Admission_Predict.csv"
df = pd.read_csv(csv_file, index_col=0)

df = df.rename(columns={
    "GRE Score":"GRE",
    "TOEFL Score": "TOEFL",
    "University Rating": "Rating",
    "Chance of Admit": "Chance"
    })

def calc_corr():
    print(df.corr())
    df.corr().to_csv('corr.csv')

def plot_top_3_attrs():
    df2 = df[["GRE", "TOEFL", "CGPA", "Chance"]]
    sns.pairplot(df2)
    plt.show()

def linear_regression_1(x, y):

    # 80% train, 20% test
    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
    scalerX = MinMaxScaler(feature_range=(0, 1))

    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    x_all = x.values.reshape(-1, 1)

    x_train = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)
    x_all = scalerX.transform(x_all)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_head_lr = lr.predict(x_test)

    from sklearn.metrics import r2_score
    print("r_square score: ", r2_score(y_test,y_head_lr))

    y_head_lr_train = lr.predict(x_train)
    print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))

    print("Coef", lr.coef_)
    print("bias", lr.intercept_)

    frame = df[["GRE", "Chance"]]
    plt.scatter(frame["GRE"], frame["Chance"], s=40, alpha=1, edgecolors='w')
    plt.xlabel("GRE")
    plt.ylabel("Chance")
    plt.plot(frame["GRE"], lr.predict(x_all), '-r')

    plt.show()

def linear_regression_2(x, y):
    # 80% train, 20% test
    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
    scalerX = MinMaxScaler(feature_range=(0, 1))

    x_train[:] = scalerX.fit_transform(x_train[:])
    x_test[:] = scalerX.transform(x_test[:])
    x_all = scalerX.transform(x)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_head_lr = lr.predict(x_test)

    print("Coef", lr.coef_)
    print("bias", lr.intercept_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x["GRE"], ys=x["CGPA"], zs=y , s=40, alpha=1, edgecolors='w', color='r')

    ax.set_xlabel('GRE')
    ax.set_ylabel('CGPA')
    ax.set_zlabel('Chance')
    # plt.plot(frame["GRE"], lr.predict(x_all), '-r')
    X = x["GRE"]
    Y = x["CGPA"]
    Z = lr.predict(x_all)
    ax.plot_trisurf(X, Y, Z)

    plt.show()
    pass

def linear_regression_all():
    y = df["Chance"]
    x = df.drop(["Chance"],axis=1)

    # 80% train, 20% test
    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
    scalerX = MinMaxScaler(feature_range=(0, 1))

    x_train[:] = scalerX.fit_transform(x_train[:])
    x_test[:] = scalerX.transform(x_test[:])

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_head_lr = lr.predict(x_test)

    print("real value of y_test[1]: " + str(y_test.iloc[1]) + " -> the predict: " + str(lr.predict(x_test.iloc[[1],:])))
    print("real value of y_test[2]: " + str(y_test.iloc[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:])))

    from sklearn.metrics import r2_score
    print("r_square score: ", r2_score(y_test,y_head_lr))

    y_head_lr_train = lr.predict(x_train)
    print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))


if __name__ == "__main__":
    # calc_corr()
    # plot_top_3_attrs()

    linear_regression_1(df["GRE"], df["Chance"])
    # linear_regression_2(df[["GRE", "CGPA"]], df["Chance"])

# sns.pairplot(df)

# plt.show()