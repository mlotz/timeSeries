import numpy as np
def batch_generator(batch_size, sequence_length, num_train, x_train_scaled, y_train_scaled):
	while True:
		x_shape = (batch_size, sequence_length, 1)
		x_batch = np.zeros(shape=x_shape, dtype=np.float64)

		y_shape = (batch_size, sequence_length, 1)
		y_batch = np.zeros(shape=y_shape, dtype=np.float64)
		for i in range(batch_size):
			idx = np.random.randint(num_train - sequence_length)
            		x_batch[i] = x_train_scaled[idx:idx+sequence_length]
			y_batch[i] = y_train_scaled[idx:idx+sequence_length]
		yield (x_batch, y_batch)
