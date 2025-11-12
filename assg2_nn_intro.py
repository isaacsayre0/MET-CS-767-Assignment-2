
import tensorflow as tf

"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Visualize: https://www.tensorflow.org/datasets/catalog/mnist Convert the samples from integers to floating-point numbers:"""

# INTENT: Load the MNIST data and show it, raw, on the monitor

mnist = tf.keras.datasets.mnist # a data set known to Keras/TensorFlow
# mnist.load_data() produces a pair of input/output tensors: one for training
# and one for testing.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("===========y_train===========")
print(tf.shape(y_train))  # format of output data: 60K records 0-9
print(y_train)  # for inspection
x_train, x_test = x_train / 255.0, x_test / 255.0  # scale down input
print("===========x_train===========")
print(tf.shape(x_train))
print("===========x_train element 0 (28 rows of 28 gray values)===========")
print(x_train[0])  # for inspection

"""Build the `tf.keras.Sequential` model (object) by stacking layers.

https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
"""

model = tf.keras.models.Sequential([  # layer format for the neural net
  # each pixel (grayscale value) mapped to one of 784 nodes
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # fully connected to hidden layer with relu
  tf.keras.layers.Dense(128, activation='relu'),
  # Dropout layer randomly sets its input units to 0 at 20% rate at each training step
  # The other inputs are scaled up by 1/0.8 so sum over all inputs is unchanged
  # Illustrative figure: http://laid.delanover.com/wp-content/uploads/2018/02/dropout.png
  # (Doc: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10) # e.g., output #7 = degree to which the output is 7
])

"""The model returns a vector of "[logits](https://developers.google.com/machine-learning/glossary#logits)" scores, one for each class--"raw, unnormalized values that will eventually be used to calculate probabilities." In other words, a tensor where the highest value indicates the most likely output.

https://en.wikipedia.org/wiki/Logit

model() acts like a function, returning the neural net object.
"""

print("===========first element of x_train (29 rows)===========")
print(x_train[:1])

predictions = model(x_train[:1]).numpy() # numpy() converts the tensor output
print("===========untrained output of first training set input===========")
print(predictions)  # expect random output, probably not favoring one digit

"""The `tf.nn.softmax` function converts these logits to "probabilities" for each class (summing to 1)."""

tf.nn.softmax(predictions).numpy()

"""Note: It is possible to bake this `tf.nn.softmax` in as the activation function for the last layer of the network. While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to
provide an exact and numerically stable loss calculation for all models when using a softmax output.

The `losses.SparseCategoricalCrossentropy` loss takes a vector of logits and a `True` index and returns a scalar loss for each example. (https://docs.google.com/document/d/16v7AIXuwAwdOZxorwSDXKRGzJA8jUYAR_D-QJTxEiyA/edit?usp=sharing) We give it a short name.
"""

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

"""This loss is equal to the negative log probability of the true class. It is zero if the model is sure of the correct class.

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to `-tf.math.log(1/10) ~= 2.3`.

Recall that y_train is the target (desired output) tensor. We select the first element.
"""

loss_fn(y_train[:1], predictions).numpy()

# Put together the NN with training process, loss, and means of evaluation.
# 'accuracy' = proportion of correct predictions vs. total number of cases
model.compile(optimizer='adam',  # a process (we'll describe) for adjusting weights
              loss=loss_fn,
              metrics=['accuracy'])  # how we'll evaluate the epochs (iterations)

"""The `Model.fit` method adjusts the model's parameters to minimize the loss:"""

# Train the net with (just) 5 epochs
model.fit(x_train, y_train, epochs=5)

"""The `Model.evaluate` method checks the model's performance, usually on a "[Validation-set](https://developers.google.com/machine-learning/glossary#validation-set)" or "[Test-set](https://developers.google.com/machine-learning/glossary#test-set)"."""

# Accuracy = fraction of correct test pairs
model.evaluate(x_test,  y_test, verbose=2)

"""The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
"""

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print("===========y_test element 0===========")
print(y_test[0])  # for inspection
probability_model(x_test[:5])  # on first 5 input images in test set