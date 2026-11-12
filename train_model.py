import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing  import LabelEncoder,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = sns.load_dataset('iris')

X = df.drop('species', axis=1)
y = df[['species']]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X_train.to_numpy())

y_train_cat = to_categorical(y_train,num_classes=3)
y_test_cat = to_categorical(y_test)

model = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    normalizer,
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(3,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train,y_train_cat,epochs=100,batch_size=4,validation_split=0.2)

loss,accuracy = model.evaluate(X_test,y_test_cat)
print(f"The Loss {loss} and Accuracy {accuracy}")

model.save("iris_model.keras")
print("Model Saved Successfully")

print("Classes_Names:",encoder.classes_)