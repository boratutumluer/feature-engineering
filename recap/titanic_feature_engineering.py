import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

sys.path.append("pythonProject/DSMLBC11/06-Feature Engineering/recap/")
from helpers.data_prep import *
from helpers.eda import *
from helpers.pandas_options import set_pandas_options

set_pandas_options(width=200)

df = pd.read_csv("pythonProject/DSMLBC11/06-Feature Engineering/recap/datasets/titanic.csv")

##################################
# 1 - FEATURE INTERACTIONS
##################################
df.columns = [col.upper() for col in df.columns]
# binary features
df["NEW_CABIN_BOOL"] = np.where(df["CABIN"].isnull(), 1, 0)
df["NEW_IS_ALONE"] = np.where(df["SIBSP"] + df["PARCH"] > 0, "NO", "YES")
# text features
df["NEW_NAME_COUNT"] = [len(letter) for letter in df["NAME"]]
df["NEW_NAME_WORD_COUNT"] = [len(letter.split(" ")) for letter in df["NAME"]]
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split(" ") if x.startswith("Dr")]))
# regex features
df["NEW_TITLE"] = df["NAME"].str.extract(' ([A-Za-z]+)\.', expand=False)
# feature interactions
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = 'young'
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = 'mature'
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] = 'senior'
df["NEW_SEX_CAT"] = df["NEW_AGE_CAT"] + df["SEX"]
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

##################################
# 2 -       OUTLIERS
##################################
removes = ["PASSENGERID", "SURVIVED"]
num_cols = [col for col in df.columns if df[col].dtypes != "O" and col not in removes]
num_but_cat = [col for col in df.columns if df[col].dtypes != "O" and col not in removes and df[col].nunique() < 10]
num_cols = list(set(num_cols) - set(num_but_cat))

for col in num_cols:
    state = check_outliers(dataframe=df, column=col)
    print(state)

for col in num_cols:
    replace_with_threshold(dataframe=df, column=col)

##################################
# 3 -   MISSING VALUES
##################################
df.isnull().sum()
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_but_car = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() > 20]
cat_cols = list(set(cat_cols) - set(cat_but_car))
cat_cols = cat_cols + num_but_cat

cat_na_cols = [col for col in df.columns if df[col].isnull().any() and col in cat_cols]
# ['EMBARKED', 'NEW_AGE_CAT', 'NEW_SEX_CAT']
num_na_cols = [col for col in df.columns if df[col].isnull().any() and col in num_cols]
# ['AGE', 'NEW_AGE_PCLASS']
cat_but_car_na_cols = [col for col in df.columns if df[col].isnull().any() and col in cat_but_car]
# ['CABIN']

missing_values_table(df)
#                 n_miss  ratio
# CABIN              687  77.10
# NEW_AGE_CAT        181  20.31
# NEW_SEX_CAT        181  20.31
# AGE                177  19.87
# NEW_AGE_PCLASS     177  19.87
# EMBARKED             2   0.22

removes_col = ["NAME", "TICKET", "CABIN"]
# We have created new features through removes_col, we can delete them and get rid of the ones with na values.
df.drop(removes_col, inplace=True, axis=1)
missing_values_table(dataframe=df)
#                 n_miss  ratio
# NEW_AGE_CAT        181  20.31
# NEW_SEX_CAT        181  20.31
# AGE                177  19.87
# NEW_AGE_PCLASS     177  19.87
# EMBARKED             2   0.22

df.groupby("NEW_TITLE").agg({"AGE": ["count", "median"]})
#             AGE
#           count   median
# NEW_TITLE
# Capt          1  64.8125
# Col           2  58.0000
# Countess      1  33.0000
# Don           1  40.0000
# Dr            6  46.5000
# Jonkheer      1  38.0000
# Lady          1  48.0000
# Major         2  48.5000
# Master       36   3.5000
# Miss        146  21.0000
# Mlle          2  24.0000
# Mme           1  24.0000
# Mr          398  30.0000
# Mrs         108  35.0000
# Ms            1  28.0000
# Rev           6  46.5000
# Sir           1  49.0000

# The null values in the age variable were filled with the median values of the new title variables we created.
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
# we would recreate other variables that contain null values depending on the age variable
df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = 'young'
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = 'mature'
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] = 'senior'
df["NEW_SEX_CAT"] = df["NEW_AGE_CAT"] + df["SEX"]
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

#              n_miss  ratio
# EMBARKED          2   0.22

# programmatically filling non-cardinal categorical variables with mode:
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x)

# difference between nunique() and len(unique())
df["EMBARKED"].nunique()  # 3, ignores null values.
len(df["EMBARKED"].unique())  # 4, also takes a null value.

##################################
# 3 -   LABEL ENCODER
##################################

# detecting binary columns
binary_col = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
# ['SEX', 'NEW_IS_ALONE'] -- female 0, male 1 | "NO" 0 "YES" 1

for col in binary_col:
    label_encoder(dataframe=df, binary_categorical_col=col)
df[binary_col].head()
df.head()
#    SEX  NEW_IS_ALONE
# 0    1             0
# 1    0             0
# 2    0             1
# 3    0             0
# 4    1             1


##################################
# 4 -     RARE ENCODER
##################################
rare_analyser(dataframe=df, target="SURVIVED", cat_cols=cat_cols)
#           COUNT     RATIO  TARGET_MEAN
# Capt          1  0.001122     0.000000
# Col           2  0.002245     0.500000
# Countess      1  0.001122     1.000000
# Don           1  0.001122     0.000000
# Dr            7  0.007856     0.428571
# Jonkheer      1  0.001122     0.000000
# Lady          1  0.001122     1.000000
# Major         2  0.002245     0.500000
# Master       40  0.044893     0.575000
# Miss        182  0.204265     0.697802
df = rare_encoder(df, 0.01)  # We considered the observations with a ratio of less than 0.01 as rare.
df["NEW_TITLE"].value_counts()
# Mr        517
# Miss      182
# Mrs       125
# Master     40
# Rare       27

##################################
# 5 -    ONE-HOT ENCODER
##################################
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(dataframe=df, categorical_col=ohe_cols, drop_first=True)

# NEW_AGE_CAT_senior  NEW_AGE_CAT_young  NEW_SEX_CAT_maturefemale  NEW_SEX_CAT_maturemale  NEW_SEX_CAT_seniormale
#         0                  0                         0                       1                       0
#         0                  0                         1                       0                       0
#         0                  0                         1                       0                       0
#         0                  0                         1                       0                       0
#         0                  0                         0                       0                       1


cat_cols, num_cols, cat_but_car = grab_col_names(dataframe=df)
rare_analyser(dataframe=df, target="SURVIVED", cat_cols=cat_cols)
# NEW_FAMILY_SIZE_11 : 2
#    COUNT     RATIO  TARGET_MEAN
# 0    884  0.992144     0.386878
# 1      7  0.007856     0.000000

# After the one-hot encoder operation, the rare encoder is made again.
# Variables with a small number of classes can be deleted because they are meaningless
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any()]
# ['SIBSP_5',
#  'SIBSP_8',
#  'PARCH_3',
#  'PARCH_4',
#  'PARCH_5',
#  'PARCH_6',
#  'NEW_NAME_WORD_COUNT_9',
#  'NEW_NAME_WORD_COUNT_14',
#  'NEW_FAMILY_SIZE_8',
#  'NEW_FAMILY_SIZE_11']
df.drop(useless_cols, axis=1, inplace=True)

df.head()

##################################
#           SUMMARY
##################################
# Generating new classes from variables that may be meaningful to the model.
# We detected and suppressed outliers that could affect linear models.
# We filled the missing variables by following various methods.
# We applied the laber encoder method on variables with binary feature for machine learning algorithms
# We applied the rare encoder method on  variables with relatively less frequency.
# With the One-Hot Encoder method, we encode variables less than 10 so that the number of classes is not more than two and the cardinality is not high.
# Finally, we removed the variables that might be useless from the data set by applied the rare analysis again.

# Our data is ready for machine learning. All calculations are based on comments, may vary.
