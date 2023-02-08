# evaluate the deep model on the test dataset
import tensorflow
from keras.models import load_model
from keras.utils import to_categorical

# load dataset
(trainX, trainY), (testX, testY) = tensorflow.keras.datasets.fashion_mnist.load_data()
# reshape dataset to have a single channel
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# one hot encode target values
testY = to_categorical(testY)

# convert from integers to floats
testX = testX.astype('float32')
# normalize to range 0-1
testX = testX / 255.0
# return normalized images

# load model
model = load_model('model.h5')
# evaluate model on test dataset
_, acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % (acc * 100.0))