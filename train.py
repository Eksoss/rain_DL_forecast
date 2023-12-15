import src.callbacks as callbacks
import src.losses as losses
import src.dataset as dataset
import src.models as models

x_train, y_train, x_test, y_test, weights, times_train, times_test = dataset.load_data()

model = models.gen_model((None, x_train.shape[-1]))
model.compile('adam', ['mse', losses.WeightedSCCE(weights, from_logits=True)])

model.fit(
    x_train, 
    y_train, 
    epochs=100, 
    validation_split=0.2, 
    callbacks=[
        callbacks.Plot(x_train[:10], y_train[:10], times_train[:10], 'train'), 
        callbacks.Plot(x_test[:10], y_test[:10], times_test[:10], 'test')
    ]
)

model.evaluate(x_test, y_test)