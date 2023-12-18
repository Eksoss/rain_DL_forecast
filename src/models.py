import tensorflow as tf

from tensorflow.keras import layers, models

def gen_model_v1(input_shape=(None, 3 + 2)):
    '''
    Basic hard-coded model for 3 step forecast, first try using an tanh 
    activation before the usage of LSTM, as to simulate how the LSTMCell sees
    a information input, but such assumption was incorrect due to the usage of 
    sigmoid internally the LSTMCell which reduces the value of 'raw' tanh, 
    rendering the artificial tanh input wrong.
    
    Also, when training both value and probabilities, the backpropagation for
    the probabilities should be restricted only to the last Dense
    trainable_variables.
    
    Returns model that takes (N, channels) and outputs [(N + 3, 1), (N + 3, 2)]
    which are values and categorical, so it can be checked the amount of rain
    and probability of rainfall on those days.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple indicating the number os timesteps and number of channels the
        model will be built with.
        
    Returns
    -------
    model : tensorflow.keras.models.Model
        A keras Model that takes the input layer with the given input_shape and
        two outputs, the first is the value, without activation and the second
        is a for categorical output with logits, so also without activation.
        
    '''
    
    lstm_cell = layers.LSTMCell(32)
    lstm_rnn  = layers.RNN(lstm_cell, return_sequences=True, return_state=True)

    _in = layers.Input(input_shape)

    c = layers.BatchNormalization()(_in)

    # abstração inicial
    c = layers.Dense(32)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Activation('tanh')(c)

    out0, *state_warm = lstm_rnn(c)

    out1, state_out1 = lstm_cell(out0[:, -1], states=state_warm)
    out2, state_out2 = lstm_cell(out1, states=state_out1)
    out3, _          = lstm_cell(out2, states=state_out2)

    pred_outs = tf.stack([out1, out2, out3], axis=1)
    outs = layers.Concatenate(axis=1)([out0, pred_outs])

    out_value = layers.Dense(1, name='value')(outs)
    out_prob  = layers.Dense(2, name='prob')(outs)

    model = models.Model(_in, [out_value, out_prob])
    return model
    
    
def gen_model_v2(input_shape=(None, 3 + 2)):
    '''
    A bit less basic hard-coded model for 3 step forecast, correcting the
    tanh trial for a LSTM layer that will create values similar to the next
    RNN output, therefore helping the forecast used by the 3 LSTMCell's.
    
    Another layer of Dense was added before the final outputs to help in the
    decision making, just in case the raw LSTM isn't enough.
    
    Returns model that takes (N, channels) and outputs [(N + 3, 1), (N + 3, 2)]
    which are values and categorical, so it can be checked the amount of rain
    and probability of rainfall on those days.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple indicating the number os timesteps and number of channels the
        model will be built with.
        
    Returns
    -------
    model : tensorflow.keras.models.Model
        A keras Model that takes the input layer with the given input_shape and
        two outputs, the first is the value, without activation and the second
        is a for categorical output with logits, so also without activation.
        
    '''
    
    lstm_cell = layers.LSTMCell(64)
    lstm_rnn  = layers.RNN(lstm_cell, return_sequences=True, return_state=True)

    _in = layers.Input(input_shape)
    c = layers.BatchNormalization()(_in)

    # abstração inicial - "Xception-like" - original com Conv2D e SeparableConv2D
    c = layers.Dense(32)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Activation('relu')(c)

    c = layers.Dense(64)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Activation('relu')(c)

    for _ in range(8):
        res = c
        
        for _ in range(3):
            c = layers.Activation('relu')(c)
            c = layers.Dense(64)(c)
            c = layers.BatchNormalization()(c)
            
        c = layers.Add()([c, res])

    # LSTM convencional, para a próxima camada conseguir usar os inputs
    # de forma consistente "tanh * sigmoid" nos steps futuros
    c = layers.LSTM(64, return_sequences=True)(c)
    out0, *state_warm = lstm_rnn(c)

    out1, state_out1 = lstm_cell(out0[:, -1], states=state_warm)
    out2, state_out2 = lstm_cell(out1, states=state_out1)
    out3, _          = lstm_cell(out2, states=state_out2)

    pred_outs = tf.stack([out1, out2, out3], axis=1)
    outs = layers.Concatenate(axis=1)([out0, pred_outs])

    out_value = layers.Dense(128)(outs)
    out_value = layers.BatchNormalization()(out_value)
    out_value = layers.Activation('relu')(out_value)
    out_value = layers.Dense(1, name='value')(out_value)
    
    out_prob = layers.Dense(128)(outs)
    out_prob = layers.BatchNormalization()(out_prob)
    out_prob = layers.Activation('relu')(out_prob)
    out_prob  = layers.Dense(2, name='prob')(out_prob)

    model = models.Model(_in, [out_value, out_prob])
    return model
    
    
def gen_model_v3(input_shape=(None, 3 + 2), era5_shape=(181, 181, 1), goes_shape=(181, 181, 1)):
    '''
    Need to reduce the goes resolution to a usable (memory-wise, like 0.25 degree), making use of a (181, 181) image
    and a kernel of 61, to simulate one day of transportation or information, may not be ideal.
    
    The kernel size comes from 800km (@10m/s) divided by average 30km per pixel, giving one day of travel of the
    signal in a horizontal radius. Not considering corners of the kernel that will have greater distances.
    
    Both era5 and goes will not have time, they'll only be used at the start of the forecast steps.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple indicating the number os timesteps and number of channels the
        model will be built with.
    era5_shape : tuple
        Tuple indicating size of era5 grid data and channels.
    goes_shape : tuple
        Tuple indicating size of goes grid data and channels.
        
    Returns
    -------
    model : tensorflow.keras.models.Model
        A keras Model that takes the input layer with the given input_shape and
        two outputs, the first is the value, without activation and the second
        is a for categorical output with logits, so also without activation.    
        
    '''
    
    # era5 conv layers
    era5_conv = models.Sequential(layers.SeparableConv2D(64, 61, 1, 'same', input_shape=(*era5_shape[:-1], 64)))
    era5_conv.add(layers.BatchNormalization())
    era5_conv.add(layers.Activation('relu'))
    # goes conv layers
    goes_conv = models.Sequential(layers.SeparableConv2D(64, 61, 1, 'same', input_shape=(*goes_shape[:-1], 64)))
    goes_conv.add(layers.BatchNormalization())
    goes_conv.add(layers.Activation('relu'))
    # before lstmcell usage
    get_center = layers.Lambda(lambda x: x[:, tf.shape(x)[1] // 2, tf.shape(x)[2] // 2, :]) # (b, i, j, c) -> (b, c)
    decision_attention = layers.MultiHeadAttention(4, 4, attention_axes=(1,))
    # lstm forecast stuff
    lstm_cell = layers.LSTMCell(64)
    lstm_rnn  = layers.RNN(lstm_cell, return_sequences=True, return_state=True)

    _in = layers.Input(input_shape)
    c = layers.BatchNormalization()(_in)
    
    # era5
    _in_era5 = layers.Input(era5_shape)
    era5_info = layers.BatchNormalization()(_in_era5)
    
    era5_info = layers.Dense(32)(era5_info)
    era5_info = layers.BatchNormalization()(era5_info)
    era5_info = layers.Activation('relu')(era5_info)
    
    era5_info = layers.Dense(64)(era5_info)
    era5_info = layers.BatchNormalization()(era5_info)
    era5_info = layers.Activation('relu')(era5_info)

    # goes
    _in_goes = layers.Input(goes_shape)
    goes_info = layers.BatchNormalization()(_in_goes)
    
    goes_info = layers.Dense(32)(goes_info)
    goes_info = layers.BatchNormalization()(goes_info)
    goes_info = layers.Activation('relu')(goes_info)
    
    goes_info = layers.Dense(64)(goes_info)
    goes_info = layers.BatchNormalization()(goes_info)
    goes_info = layers.Activation('relu')(goes_info)
    
    # abstração inicial - "Xception-like" - original com Conv2D e SeparableConv2D
    c = layers.Dense(32)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Activation('relu')(c)

    c = layers.Dense(64)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Activation('relu')(c)

    for _ in range(8):
        res = c
        
        for _ in range(3):
            c = layers.Activation('relu')(c)
            c = layers.Dense(64)(c)
            c = layers.BatchNormalization()(c)
            
        c = layers.Add()([c, res])

    # LSTM convencional, para a próxima camada conseguir usar os inputs
    # de forma consistente "tanh * sigmoid" nos steps futuros
    c = layers.LSTM(64, return_sequences=True)(c)
    out0, *state_warm = lstm_rnn(c)

    last_out = out0[:, -1]
    last_state = state_warm
    pred_outs = []
    for _ in range(3):
        era5_info = era5_conv(era5_info)
        era5_center = get_center(era5_info)
        
        goes_info = era5_conv(goes_info)
        goes_center = get_center(goes_info)
        
        space_info = layers.Concatenate()([last_out, era5_center, goes_center])
        
        decided_space_info = decision_attention(last_out, space_info)
        decided_space_info = layers.Add()([last_out, decided_space_info])
        decided_space_info = layers.BatchNormalization()(decided_space_info)
        decided_space_info = layers.Activation('tanh')(decided_space_info)
        # this breaks the proposal of tanh * sigmoid from the last LSTM
        
        last_out, last_state = lstm_cell(decided_space_info, states=last_state)
        pred_outs.append(last_out)

    pred_outs = tf.stack(pred_outs, axis=1)
    outs = layers.Concatenate(axis=1)([out0, pred_outs])

    out_value = layers.Dense(128)(outs)
    out_value = layers.BatchNormalization()(out_value)
    out_value = layers.Activation('relu')(out_value)
    out_value = layers.Dense(1, name='value')(out_value)
    
    out_prob = layers.Dense(128)(outs)
    out_prob = layers.BatchNormalization()(out_prob)
    out_prob = layers.Activation('relu')(out_prob)
    out_prob  = layers.Dense(2, name='prob')(out_prob)

    model = models.Model([_in, _in_era5, _in_goes], [out_value, out_prob])
    return model
    
    
if __name__ == '__main__': # testing only
    import numpy as np
    
    a = np.random.rand(32, 10, 9)
    b = np.random.rand(32, 181, 181, 5)
    c = np.random.rand(32, 181, 181, 1)
    
    d = np.random.rand(32, 13, 1)
    e = np.random.randint(2, size=(32, 13, 1))
    
    model = gen_model_v3((None, 9), (181, 181, 5), (181, 181, 1))
    model.compile('adam', ['mse', 'sparse_categorical_crossentropy'])
    
    model.fit([a, b, c], [d, e], epochs=1) # worked :D
    # res = model.predict([a, b, c])
    # print(res[0].shape, res[1].shape) # worked :D