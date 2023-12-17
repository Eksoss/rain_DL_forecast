import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks

class Plot(callbacks.Callback):
    def __init__(self, x, y, times, name=''):
        super().__init__()
        self.x = x
        self.y_plot = np.exp(y[0][..., 0]) - 1.
        self.y_bin_plot = y[1][..., 0]
        self.times = times
        self.name = name
        
    def on_epoch_end(self, epoch, losses):
        if epoch % 5 == 0:
            yhat, yhat_bin = self.model.predict(self.x)
            
            yhat_plot = np.exp(yhat[..., 0]) - 1.
            yhat_bin_plot = tf.nn.softmax(yhat_bin).numpy()[..., 1]
            
            _max_ylim = max(yhat_plot.max(), self.y_plot.max())
            
            for idx, (yp, ybp, date) in enumerate(zip(yhat_plot, yhat_bin_plot, self.times)):
                
                fig, axs = plt.subplots(2, 1, figsize=(20., 10.))
                
                # plot values
                axs[0].plot(self.y_plot[idx].clip(0., None), 'bo-', label='obs')
                axs[0].plot(yp.clip(0., None), 'ro-', label='frc')
                axs[0].set_title('prec | ' + date.strftime('%Y-%m-%d'))
                axs[0].set_ylim(0, _max_ylim)
                axs[0].legend()
                
                # plot bin
                axs[1].plot(self.y_bin_plot[idx], 'bo-', label='obs')
                axs[1].plot(ybp, 'ro-', label='frc')
                axs[1].set_title('bin | ' + date.strftime('%Y-%m-%d'))
                axs[1].set_ylim(0, 1)
                axs[1].legend()
                
                plt.savefig(f'imgs/plot_{self.name}_epoch_%03d_%02d.png'%(epoch, idx))
                plt.close()