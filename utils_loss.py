import keras.backend as K


def MSE_BCE(y_true, y_pred, alpha=1000, beta=10):
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return alpha * mse + beta * bce
