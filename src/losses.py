import tensorflow as tf
from tensorflow.keras import losses

class WeightedSCCE(losses.Loss):
    '''
    Weighted Sparse Categorical CrossEntropy
    
    '''
    
    def __init__(self, class_weight, from_logits=False, ignore_class=None, name='weighted_scce', **kwargs):
        self.class_weight = tf.convert_to_tensor(class_weight,
            dtype=tf.float32)

        self.name = name
        self.reduction = losses.Reduction.NONE
        self.unreduced_scce = losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, ignore_class=ignore_class, name=name,
            reduction=self.reduction)

    def get_config(self,):
        config = super().get_config()
        config.update(
            {
                'reduction': self.reduction,
                'name': self.name,
                'unreduced_scce': self.unreduced_scce,
                'class_weight': self.class_weight.numpy(), # as tensor tf 2.10 bugs when trying to serialize with JSON
            }
        )
        return config
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.cast(self.unreduced_scce(y_true, y_pred, sample_weight), 'float32')
        if self.class_weight is not None:
            weight_mask = tf.cast(tf.gather(self.class_weight, tf.cast(tf.clip_by_value(y_true[..., 0], 0, 1), 'int32')), 'float32')
            loss = tf.math.multiply(loss, weight_mask)
        return tf.reduce_mean(loss)