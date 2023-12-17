import tensorflow as tf

from tensorflow.keras import layers, models

def gen_model_v1(input_shape=(None, 3 + 2)):
    '''
    Basic hard-coded model for 3 step forecast, first try using an tanh activation
    before the usage of LSTM, as to simulate how the LSTMCell sees a information
    input, but such assumption was incorrect due to the usage of sigmoid internally
    the LSTMCell which reduces the value of 'raw' tanh, rendering the artificial
    tanh input wrong.
    
    Also, when training both value and probabilities, the backpropagation for
    the probabilities should be restricted only to the last Dense trainable_variables.
    
    Returns model that takes (N, channels) and outputs [(N + 3, 1), (N + 3, 2)]
    which are values and categorical, so it can be checked the amount of rain
    and probability of rainfall on those days.
    
    Parameters
    ----------
    input_shape : tuple
        Tuple indicating the number os timesteps and number of channels the model
        will be built with.
        
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