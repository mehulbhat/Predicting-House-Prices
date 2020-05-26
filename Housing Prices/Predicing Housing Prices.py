import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

%matplotlib inline
tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv('data.csv', names = column_names)

#visualizing dataframe
df.head()

#checking for missing data
df.isna().sum()

#Data normalization
df = df.iloc[:, 1:]
df_norm = (df - df.mean())/df.std()
df_norm.head()

#Convering label values back to original after normalization
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

#selecting features and visualizing
x= df_norm.iloc[:, :6]
x.head()

#selecting labels and visualizing
y = df_norm.iloc[:, -1]
y.head()

#Feature and label values
x_arr = x.values
y_arr = y.values

#Visualizing the shape
print('features array: ', x_arr.shape)
print('label array: ', y_arr.shape)

#Train and test split
X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size = 0.05,
                                                    random_state = 0)
#visualizing shape
print(X_train.shape)
print(X_test.shape)

#creating model with defined architecture
def get_model():
    model = Sequential([
        Dense(10, input_shape = (6,), activation='relu'),
        Dense(20, activation='relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])
    model.compile(
    loss='mse',
    optimizer='adam'
    )
    return model
get_model().summary()

#model training
es_cb = EarlyStopping(monitor='val_loss', patience=5)

model = get_model()
preds_on_untrained = model.predict(X_test)

history = model.fit (
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 100,
    callbacks = [es_cb]
)

#Plotting training and validation loss
plot_loss(history)

#Plotting raw predictions
preds_on_trained = model.predict(X_test)
compare_predictions(preds_on_untrained, preds_on_trained, y_test)

#Plotting price predictions after converting back to original values
price_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_trained = [convert_label_value(y) for y in preds_on_trained]
price_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_untrained, price_trained, price_test)