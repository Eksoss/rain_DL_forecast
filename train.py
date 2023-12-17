import src.callbacks as callbacks
import src.losses as losses
import src.dataset as dataset
import src.models as models

x_train, y_train, x_test, y_test, weights, times_train, times_test = dataset.load_data_v2()

model = models.gen_model_v2((None, x_train.shape[-1]))
model.compile('adam', ['mse', losses.WeightedSCCE(weights, from_logits=True)])

model.fit(
    x_train, 
    y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        callbacks.Plot(x_train[:20], [y_train[0][:20], y_train[1][:20]], times_train[:20], 'train'), 
        callbacks.Plot(x_train[365:365+20], [y_train[0][365:365+20], y_train[1][365:365+20]], times_train[365:365+20], 'train2'), 
        callbacks.Plot(x_test[:20], [y_test[0][:20], y_test[1][:20]], times_test[:20], 'test'),
        callbacks.Plot(x_test[365:365+20], [y_test[0][365:365+20], y_test[1][365:365+20]], times_test[365:365+20], 'test2'),
    ]
)

model.evaluate(x_test, y_test)

model.save('models/model_v2.h5')