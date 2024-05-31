
import json
import keras
from keras import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


data_path = "data_json"


def load_data(data_path):
    print("Загрузка данных\n")
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Загруженные данные")

    return x, y


def prepare_datasets(test_size, val_size):
    # Загрузка данных
    x, y = load_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size)

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(input_shape):
    model = Sequential()

    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_datasets(0.3, 0.25)

    #print(x_train.shape[0])

    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    # Сборка модели

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Тренировака модели на обучающей выборке
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=25)

    # plot accuracy/error for training and validation
    # plot_history(history)
    fig, axs = plt.subplots(2, figsize=(10, 10))
    # accuracy
    axs[0].plot(history.history["accuracy"], label="train")
    axs[0].plot(history.history["val_accuracy"], label="test")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].set_title("Accuracy")

    ## Error
    axs[1].plot(history.history["loss"], label="train")
    axs[1].plot(history.history["val_loss"], label="test")
    axs[1].set_ylabel("Error")
    axs[1].legend()
    axs[1].set_title("Error")
    plt.show()
    # Проверка модели на тестовой выборке
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save("model_LSTM.keras")
    print("Модель сохранена")



