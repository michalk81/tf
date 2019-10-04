

import random
import pydot
import graphviz
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def random_process(coeff,drift,n=500):
    random_walk = list()
    random_walk.append(-1 if random.random() < 0.5 else 1)
    for i in range(1, n):
        movement = -1 if random.random() < 0.5 else 1
        value = coeff * random_walk[i - 1] + movement + drift * i
        random_walk.append(value)
    return random_walk

def main():
    '''pydot.Dot.create(pydot.Dot())'''
    print tf.__version__
    samples = list()
    for i in range(0,2000):
        samples.append((random_process(1.0,0.0),0))
    for i in range(0,2000):
        adjustCoeff = 0.88+random.random()/10.0
        samples.append((random_process(adjustCoeff,0.0),1))
    for i in range(0,2000):
        drift = 0.0001+(random.random()/10000.0)
        samples.append((random_process(1.0,drift),2))

    random.shuffle(samples)
    test = list()
    for i in range(0, 100):
        test.append((random_process(1.0, 0.0), 0))
    for i in range(0, 100):
        adjustCoeff = 0.88 + random.random() / 10.0
        test.append((random_process(adjustCoeff, 0.0), 1))
    for i in range(0, 100):
        drift = 0.0001 + (random.random() / 10000.0)
        test.append((random_process(1.0, drift), 2))

    j = 0
    classes=["RANDOM_WALK","MEAN_REVERTING","DRIFT"]

    training = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    t = K.variable(training)
    l = K.variable(labels)

    tst = K.variable([ts[0] for ts in test])
    tstLabel = K.variable([ts[1] for ts in test])

    print t
    print t.shape

    plt.figure(figsize=(10, 10))
    for i in range(25):
        j = int(random.random()*3000)
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.plot(training[j])
        plt.xlabel(classes[labels[j]])
    plt.show()

    model = keras.Sequential([
        keras.layers.Dropout(0.2, input_shape=(500,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    '''keras.utils.plot_model(model, 'model.png')'''

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(t, l, epochs=10,steps_per_epoch=100)

    # Plot training & validation accuracy values

    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.show()

    test_loss, test_acc = model.evaluate(tst, tstLabel,steps=10)



    print('\nTest accuracy:', test_acc)


main()