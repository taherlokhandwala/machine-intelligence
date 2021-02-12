import pandas as pd
data = pd.read_csv("LBW_Dataset.csv")

# substitute with mode
for i in ['Delivery phase', 'Education', 'Residence']:
    data[i].fillna(data[i].mode()[0], inplace=True)

# data.hist(column='Age')
# plt.show()
# the above showed that median is better than mode due to skew
# substitute with median due to skew
data['Age'].fillna(data['Age'].median(), inplace=True)


data['Weight'] = data['Weight'].fillna(
    data.groupby('Age')['Weight'].transform('median'))
data['HB'] = data['HB'].fillna(data.groupby('Age')['HB'].transform('mean'))
data['BP'] = data['BP'].fillna(data.groupby('Age')['BP'].transform('mean'))

data['HB'] = data['HB'].round(decimals=1)
data['BP'] = data['BP'].round(decimals=3)
data['Weight'] = data['Weight'].round(decimals=0)

# Due to the machine learning model not knowing what categorical data is
# we cannot directly use it. We therefore convert to one hot encoding to reprsent the data
columns = ["Community", "Delivery phase", "IFA", "Residence"]
data = pd.get_dummies(data, prefix_sep="_", columns=columns)


# checks for any null character
# print(data[data.isnull().sum(axis=1) > 0])

# converts to int if needed
# for i in ['Age','Delivery phase','Education','Residence','Weight']:
# 	data[i]=data[i].astype(int)
data.to_csv(r'LBW_Dataset_Cleaned.csv', index=False)
