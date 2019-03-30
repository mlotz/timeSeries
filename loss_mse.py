import tensorflow as tf

def loss_mse_warmup(y_true, y_pred):

	warmup_steps = 50
	y_true_slice = y_true[:, warmup_steps:, :]
	y_pred_slice = y_pred[:, warmup_steps:, :]
	loss = tf.losses.mean_squared_error(labels=y_true_slice,predictions=y_pred_slice)
	loss_mean = tf.reduce_mean(loss)
	return loss_mean
