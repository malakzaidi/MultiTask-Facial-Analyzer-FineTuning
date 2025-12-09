import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Test actual GPU computation
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print("GPU computation successful!")