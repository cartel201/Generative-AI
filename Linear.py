import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {'Size (sq ft)': [1500, 1800, 2400, 3000, 3500],
        'Bedrooms': [3, 4, 3, 4, 5],
        'Price': [400000, 500000, 600000, 650000, 700000]}

df = pd.DataFrame(data)

# Independent variables (features)
X = df[['Size (sq ft)', 'Bedrooms']]

# Dependent variable (target)
y = df['Price']

model = LinearRegression()
model.fit(X, y)

new_house = np.array([[2500, 3]]) 
price = model.predict(new_house)

print(f"Predicted price of the new house: ${price[0]:,.2f}")








#-----------------------------------------------------Logistic Regression---------------------------------------------------------


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data = {'Word1': [3, 1, 5, 7, 2],
        'Word2': [2, 5, 1, 6, 4],
        'Spam': [1, 0, 1, 1, 0]}  # 1 = spam, 0 = not spam

df = pd.DataFrame(data)

X = df[['Word1', 'Word2']]

y = df['Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")






#-------------------------Clusterin (K-means)-----------------------------------------------------


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {'Spending': [200, 300, 400, 500, 600, 700, 1000, 1200, 1400],
        'Frequency': [3, 2, 4, 5, 2, 4, 6, 8, 5]}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[['Spending', 'Frequency']])

# Plot the clusters
plt.scatter(df['Spending'], df['Frequency'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Spending')
plt.ylabel('Frequency')
plt.title('Customer Segmentation')
plt.show()





#--------------------





from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = {'Age': [22, 25, 47, 35, 55, 33, 66, 77, 75, 86, 100, 23, 29, 41, 51, 61, 67, 72, 81, 90, 32, 28, 60, 45, 52, 69, 80, 39, 48, 54, 59],
        'Income': [30000, 50000, 70000, 80000, 120000, 140000, 160000, 180000, 190000, 200000, 240000, 35000, 55000, 75000, 85000, 130000, 150000, 165000, 195000, 220000, 32000, 47000, 62000, 98000, 150000, 160000, 175000, 95000, 100000, 120000, 135000],
        'Purchased': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1]}  # 1 = bought, 0 = did not buy

df = pd.DataFrame(data)

X = df[['Age', 'Income']]

y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = (y_pred == y_test).mean()
print(f"Model accuracy: {accuracy * 100:.2f}%")



# Conclusion
# Linear Regression is used for predicting continuous values.
# Logistic Regression is used for classification tasks (binary or multi-class).
# K-Means Clustering is used for grouping data into clusters.
# Decision Trees help in making decisions based on splitting data into subsets.