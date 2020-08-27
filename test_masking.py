import tensorflow as tf
tf.enable_eager_execution()

label=tf.constant([3,2,2,4,5,6,1,0,0])

actual_len = tf.squeeze(tf.where(tf.equal(label, 1)))
actual_len = tf.cast(actual_len,tf.int32)

mask_num = tf.cond(tf.less(actual_len,3),
                true_fn = lambda: tf.constant([0]),
                false_fn= lambda: tf.random.uniform(shape=[1], minval=0, maxval=tf.cast(actual_len/3,tf.int32)+1, dtype=tf.int32))

indices = tf.reshape(tf.range(start=0, limit=actual_len, delta=1), [actual_len,1])
indices = tf.random.shuffle(indices)[mask_num[0]:]
indices = tf.sort(indices,axis=0)

reduced_label = tf.gather_nd(label, indices)
label = tf.concat([reduced_label, [1], tf.zeros(tf.size(label)-tf.size(reduced_label)-1,tf.int32)],axis=0)
print(label)