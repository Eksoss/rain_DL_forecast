import src.callbacks as callbacks
import src.losses as losses
import src.dataset as dataset
import src.models as models

x_train, y_train, x_test, y_test, weights, times_train, times_test = dataset.load_data_v3()

model = models.gen_model_v3((None, x_train[0].shape[-1]), (None, None, x_train[1].shape[-1]), (None, None, x_train[2].shape[-1])) # not (181, 181)
model.compile('adam', ['mse', losses.WeightedSCCE(weights, from_logits=True)])

model.fit(
    x_train, 
    y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        callbacks.Plot([x[:20] for x in x_train], [y[:20] for y in y_train], times_train[:20], 'train'), 
        # callbacks.Plot(x_train[:20], [y_train[0][:20], y_train[1][:20]], times_train[:20], 'train'), 
        # callbacks.Plot(x_train[365:365+20], [y_train[0][365:365+20], y_train[1][365:365+20]], times_train[365:365+20], 'train2'), 
        callbacks.Plot([x[:20] for x in x_test], [y[:20] for y in y_test], times_test[:20], 'test'),
        # callbacks.Plot(x_test[:20], [y_test[0][:20], y_test[1][:20]], times_test[:20], 'test'),
        # callbacks.Plot(x_test[365:365+20], [y_test[0][365:365+20], y_test[1][365:365+20]], times_test[365:365+20], 'test2'),
    ]
)

model.evaluate(x_test, y_test)

model.save('models/model_v3.h5')