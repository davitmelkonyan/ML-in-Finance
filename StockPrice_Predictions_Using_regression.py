import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras


stock_price_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Python & ML in Finance/Part 3. AI and ML in Finance/stock.csv')
stock_vol_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Python & ML in Finance/Part 3. AI and ML in Finance/stock_volume.csv")
stock_price_df = stock_price_df.sort_values(by = ['Date'])
stock_vol_df = stock_vol_df.sort_values(by = ['Date'])
stock_price_df.isnull().sum()
stock_vol_df.isnull().sum()
stock_price_df.info()
stock_vol_df.info()
stock_vol_df.describe()


def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()

interactive_plot(stock_price_df, 'Stock Prices')
def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})

def trading_window(data):
  n = 1
  data['Target'] = data[['Close']].shift(-n)
  return data

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
price_volume_target_df = trading_window(price_volume_df)
price_volume_target_df = price_volume_target_df[:-1]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns = ['Date']))

X = price_volume_target_scaled_df[:,:2]
y = price_volume_target_scaled_df[:,2:]
split = int(0.65 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

def show_plot(data, title):
  plt.figure(figsize = (13, 5))
  plt.plot(data, linewidth = 3)
  plt.title(title)
  plt.grid()

show_plot(X_train, 'Training Data')
show_plot(X_test, 'Testing Data')


from sklearn.linear_model import Ridge
# Note that Ridge regression performs linear least squares with L2 regularization.
# Create and train the Ridge Linear Regression  Model
regression_model = Ridge()
regression_model.fit(X_train, y_train)

lr_accuracy = regression_model.score(X_test, y_test)
print("Linear Regression Score: ", lr_accuracy)

predicted_prices = regression_model.predict(X)
Predicted = []
for i in predicted_prices:
  Predicted.append(i[0])

close = []
for i in price_volume_target_scaled_df:
  close.append(i[0])

df_predicted = price_volume_target_df[['Date']]
df_predicted['Close'] = close
df_predicted['Prediction'] = Predicted
interactive_plot(df_predicted, "Original Vs. Prediction")

#LSTM Series model
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')
training_data = price_volume_df.iloc[:, 1:3].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)

X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])

split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)

predicted = model.predict(X)
test_predicted = []

for i in predicted:
  test_predicted.append(i[0][0])

df_predicted = price_volume_df[1:][['Date']]
df_predicted['predictions'] = test_predicted

close = []
for i in training_set_scaled:
  close.append(i[0])

df_predicted['Close'] = close[1:]
interactive_plot(df_predicted, "Original Vs Prediction")