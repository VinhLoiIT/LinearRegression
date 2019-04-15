
#%% [markdown]
# # Linear Regression
# ------
# # Group Info
# | MSSV | Họ tên | Email |
# |------|--------|-------|
# | 1612348 | Lý Vĩnh Lợi | vinhloiit1327@gmail.com |
# | 1612336 | Vũ Thùy Linh | linhnhung1419@gmail.com |
# | 1612154 | Hoàng Hải Giang | giangietsubasa@gmail.com |

#%% [markdown]
# # Load Dataset
# Dataset được lấy từ [Kaggle - Graduate Admissions](https://www.kaggle.com/mohansacharya/graduate-admissions). Do có một vài column tên khá dài và lỗi typo nên đổi tên để dễ làm việc hơn
#%%
import pandas as pd
csv_file = "Admission_Predict.csv"
df = pd.read_csv(csv_file, index_col=0)

# Đổi tên một số cột trong dataset
df = df.rename(columns={
    "GRE Score":"GRE",
    "TOEFL Score": "TOEFL",
    "University Rating": "Rating",
    "Chance of Admit ": "Chance"
    })
# Hiển thị 5 dòng đầu của dataset
print(df.head(5))

#%% [markdown]
# # Visualize dataset
# Visualize dataset theo từng cột thuộc tính với tiên đoán để có cái nhìn tổng quan về mối quan hệ giữa từng cặp thuộc tính

#%%
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.pairplot(df)
plt.show()

#%% [markdown]
# # Chọn thuộc tính
# Dễ thấy các thuộc tính "GRE", "TOEFL", "CGPA" có mối quan hệ tuyến tính với thuộc tính tiên đoán "Chance", do đó ta chọn 3 thuộc tính này làm các biến của mô hình

#%%
df = df[["GRE", "TOEFL", "CGPA", "Chance"]]
sns.pairplot(df)
plt.show()

#%% [markdown]
# # Đặt biến
# - Đổi tên biến cho giống với mô hình tính toán lý thuyết:
#   * $z_1$ là điểm GRE
#   * $z_2$ là điểm TOEFL
#   * $z_3$ là điểm CGPA
#   * $y$ là khả năng được nhận vào trường - cột Chance
# - Khi đó mô hình hồi quy tuyến tính có dạng $y = \beta_0 +\beta_1 z_1 + \beta_2 z_2 + \beta_3 z_3$

#%% [markdown]
# # Sum of squares estimation
# $S(b) = \sum_{i=1}^{n}{(y_i - \beta_0 - \beta_1 z_1 - \beta_2 z_2 - ... - \beta_r z_r)}$
# 

#%%
