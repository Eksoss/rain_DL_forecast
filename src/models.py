import tensorflow as tf

from tensorflow.keras import layers, models

def gen_model(input_shape=(None, 3 + 2)):
    '''
    Basic hard-coded model for 3 step forecast
    '''
    lstm_cell = layers.LSTMCell(32)
    lstm_rnn  = layers.RNN(lstm_cell, return_sequences=True, return_state=True)

    _in = layers.Input(input_shape) # (None, 3 + 2)

    c = layers.BatchNormalization()(_in)

    # abstração inicial
    c = layers.Dense(32)(c)
    # usando tanh pois a saída padrão do lstm é tanh, quero manter o padrão
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
    
def main():
    import numpy as np
    a = np.random.rand(100, 30, 5)
    b = a[..., :1]
    c = np.int32(b > 0.5)
    model = gen_model()
    model.compile('adam', ['mse', 'sparse_categorical_crossentropy'])
    
    model.fit(a[:, :-3], [b, c], epochs=10)
    
main()

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)


    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
        
        
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
