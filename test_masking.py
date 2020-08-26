import tensorflow as tf
tf.enable_eager_execution()

label=tf.constant([3,2,4,5,1])

actual_len = tf.squeeze(tf.where(tf.equal(label, 1)))
print(actual_len)
mask_num = tf.cond(tf.less(actual_len,3),
                true_fn = lambda: 0,
                false_fn= lambda: tf.random.uniform(shape=[1], minval=0, maxval=tf.cast(actual_len/3,tf.int64)+1, dtype=tf.dtypes.int64))

print(mask_num)
indices = tf.reshape(tf.range(start=0, limit=actual_len, delta=1), [actual_len,1])

indices = tf.random.shuffle(indices)[:tf.squeeze(actual_len-mask_num)]
indices = tf.sort(indices,axis=0)
# mask_ind = tf.slice(indices,[0,0],[tf.squeeze(actual_len-mask_num),1])
print(indices)
reduced_label = tf.gather_nd(label, indices)
print(reduced_label)
tf.concat(1, tf.zeros([5-len(reduced_label)-1],tf.int32))
# put 1 in mask_ind, leaving the rest of positions 0
bool_mask = tf.cast(tf.scatter_nd(mask_ind,tf.ones(mask_num,tf.int32),[5]), tf.bool)
# put mask_token in mask_ind, leaving the rest of positions same as label
masked_label = tf.where(bool_mask, tf.ones(5,tf.int32)*0, label)


print(masked_label)