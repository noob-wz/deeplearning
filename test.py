import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[50], [80], [100], [120], [150]])
y = np.array([160, 240, 310, 350, 460])

model = LinearRegression()
model.fit(X, y)

new_house = np.array([[100]])
predicted_price = model.predict(new_house)

print(predicted_price)