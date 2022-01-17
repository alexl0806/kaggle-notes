from tensorflow import keras
from tensorflow.keras import layers

# easiest way to create a model in Keras is through keras.Sequential, creates a neural netowkr
# as a stack of layers.

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3]) # input shape = 3 since we're using three input features, units for how many outputs we want
])

# gets the weight and bias of the model
model.weights

# neural networks typically organize their neurons into layers, when linear units have common sets of input,
# we get a dense layer

# an activation function applies a function to each of a layer's outputs (its activations)
# we use activation functions so neural networks can learn non-linear relationships

max(0, x) # reectifier function

# ReLu function = when we attach the rectifier to a linear unit
# layers before the output layers are sometimes called hidden since we never see their outputs directly

# sequential model with relu layers
model = keras.Sequential([
    # reLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),

    # final linear output layers
    layers.Dense(units=1),
]) # make sure to pass all the layers together in a list

# alternative way to attach an activation function to a dense layer
layers.Dense(units=8),
layers.Activation('relu')

# variants of relu activation are elu, selu, swish, etc.

# loss function measures the disparity between the target's true value and the value the model predicts
# regression problems -> predict some numeral value. commmon loss function is mean absolute error
# other loss functions are mean-squared error (MSE) or Huber loss

# during training, model will use the loss function as a guide for finding the correct values of its weights

# optimizer is an algorithm that adjusts the weights to minimize the loss
# stochastic gradient descent => iterative algorithms that train a network in steps:
# 1) sample some training data and run it through the network to make predictions.
# 2) measure the loss between the predictions and the true values.
# 3) adjust the weights in a direction that makes the loss smaller
# rinse and repeat

# each iteration's sample of training data is called a minibatch, complete round is called an epoch
# number of epochs you train for is how many times the network will see each training example.

# size of shifts is determined by the learning rate. smaller learning rate means network needs to see more minibatches
# learning rate and size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds

# using Adam (SGD algorithm) for loss function and optimizer
model.compile(
    optimizer="adam",
    loss="mae"
) # can also directly aceess loss and optimizer through the Keras API to tune parameters

# then we start training
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# example
history = model.fit(
    X, y,
    batch_size=128,
    epochs=200
)

# plotting the loss
history_df = pd.DataFrame(history.history) # convert the training history to a dataframe
history_df['loss'].plot();

# You probably saw that smaller batch sizes gave noisier weight updates and loss curves. This is because each batch is a small sample of data and smaller samples tend to give noisier estimates. Smaller batches can have an "averaging" effect though which can be beneficial.

# Smaller learning rates make the updates smaller and the training takes longer to converge. Large learning rates can speed up training, but don't "settle in" to a minimum as well. When the learning rate is too large, the training can fail completely.

# information in training data: signal and noise, signal helps model make predictions from new data, noise doesn't help the model

# we can plot the learning curves (loss on data sets) of validation and training. The gap represents the amouunt of noise.

# underfitting: loss is not as low as it could be because the model hasn't learned enough signal
# overfitting: loss is not as low as it could be because the model has learned too much noise

# capacity: size and complexity of the patterns it is able to learn.
# change capacity by adding more units to existing layers, or by deepening (adding more layers)
# wider networks are better on linear relationships, deeper ones prefer no-linear relationships

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

# when a model learns too much noise, validation loss may begin to increase during training, so stop the training
# when the validation loss isn't decreasing anymore (aka early stopping)

# once we detect that the validation loss is starting to rise again, we can reset the weights back to where the minimum occurred.
# with early stopping, we use very large epochs so the network won't underfit

# we add early stopping through a callback (function that runs every so often while the network trains)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping( # early stopping runs after every epoch
    min_delta=0.001, # min amount of change to count as an improvement
    patience=20, # how many epochs to wait before stoppity stopping
    restore_best_weights=True
)

# to implement early stopping, we add it as an argument in fit
history = model.fit(
    X_train, y_train,
    validation_data=(X,valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0, # turn off training log
)

# dropout => randomly drop out some fraction of a layer's input units every step of the training => less overfitting

# adding dropout
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])

# with neural networks, it's generally a good idea to put all of your data on a common scale,
# ie: scikit-learn's StandardScaler or MinMaxScaler
# => SGD will shift the network weights in proportion to how large an activation the data produces

# batch normalization layers => looks at each batch as it comes in, first normalizing the batch with its own
# mean and standard deviation then puts data on a new scale with two trainable rescaling parameters
# we can use Batchnorm

# can put batch normalization after a layer:
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),

# between a layer and its activation function:
layers.Dense(16)
layers.BatchNormalization(),
layers.Activation('relu'),

# if added as the first layer of the network, it can act as a kind of adaptive preprocessor, standing in for
# something like Sci-kit Learn's StandardScaler

# SGD needs a loss function that changes smoothly, but accuracy changes in jumps, thus we use cross-entropy function

# to convert the real-valued outputs produced by a dense layer into probabilities, we attach the sigmoid activation function
# to get the final class prediction, we define a threshold probability

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000
    callbacks=[early_stopping],
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot() # starting from the fifth epoch
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
