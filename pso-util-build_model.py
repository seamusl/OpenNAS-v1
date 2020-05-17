# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License
# Simple utility for creating a model from a set of weights and yaml file with architecture
# not used as part of main AutoNAS application

from keras.models import model_from_yaml
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

yaml_file = open('best-gBest-model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights('best-gBest-weights.h5')

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)
loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

loaded_model.save('model_1.h5')  # save the model (overwrite weights file with full arch)

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()  # load the training and testing data.


testYc = to_categorical(testY)

print(loaded_model.summary())

_, acc = loaded_model.evaluate(testX, testYc, verbose=0)
print('Model Accuracy: %.4f' % acc)


