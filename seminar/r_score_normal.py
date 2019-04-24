import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def least_square_estimation(x:np.ndarray,y:np.ndarray):
    inv = np.linalg.inv(x.transpose().dot(x))
    return inv.dot(x.transpose()).dot(y)
    # return x.transpose().dot(x).inv().dot(x.transpose).dot(y)

seed = 42
np.random.seed(seed) # make repeate random values

n = 20
z = np.linspace(-10,10,n)
epsilon = np.random.randn(n)
y = 2*z + epsilon
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(z,y, label="real data")
ax.set_xlabel("z")
ax.set_ylabel("y")
ax.legend()
plt.show()

# df = pd.DataFrame({"z":z, "y": y})
# print(df)
# df.to_csv("data_normal.csv")

z_0 = np.ones((n))

Z = np.vstack((z_0,z)).transpose()
Y = y.transpose()
b_hat = least_square_estimation(Z,Y)
print("Beta =", b_hat)

y_plot = np.vstack((z_0, z)).transpose().dot(b_hat)
plt.plot(z,y_plot, color='r', label="y={:.02f}+{:.02f}z".format(b_hat[0], b_hat[1]))
# plt.scatter(z,y_plot, color='g')

r_squares = np.var(y_plot)/np.var(y)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
plt.text(-10,-2,"r-squares={:.02f}".format(r_squares), bbox=props)

plt.legend()
plt.xlabel("z")
plt.ylabel("y")
plt.title("Hồi quy tuyến tính 1 biến")

plt.show()