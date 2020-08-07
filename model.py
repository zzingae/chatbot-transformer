from official.transformer.model.model_params import BASE_PARAMS as Transformer_params
import official.transformer.utils.metrics as metrics
from official.utils.logs import hooks_helper
from official.transformer.model.transformer import Transformer
import tensorflow as tf
import random
import os


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    # Create a named tensor that will be logged using the logging hook.
    # The full name includes variable and names scope. In this case, the name
    # is model/get_train_op/learning_rate/learning_rate
    tf.identity(learning_rate, "learning_rate")

    return learning_rate

def model_fn(features, mode, params):
    # extend dict values to defaultdict
    _params = Transformer_params.copy()
    for k in params:
        v = params[k]
        _params[k] = v
    params = _params

    if mode == tf.estimator.ModeKeys.PREDICT: features['answer']=None

    transformer = Transformer(params, (mode == tf.estimator.ModeKeys.TRAIN))
    logits = transformer(features['question'], features['answer'])

    # estimator part
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        xentropy, weights = metrics.padded_cross_entropy_loss(logits, features['answer'], params["label_smoothing"], params["vocab_size"])
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        learning_rate = get_learning_rate(params['learning_rate'], params['hidden_size'], params['learning_rate_warmup_steps'])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=params['optimizer_adam_beta1'], beta2=params['optimizer_adam_beta2'], epsilon=params['optimizer_adam_epsilon'])

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook({"logit": tf.reduce_mean(logits), "answer": features['answer'][:5], "logitmax": tf.argmax(logits[0:5], -1)}, every_n_iter=100)
        eval_metric_ops = metrics.get_eval_metrics(logits, features['answer'], params)
        tensors_to_log = {}
        for k in eval_metric_ops:
            tensors_to_log[k.split('/')[-1]] = eval_metric_ops[k][1].name
            tf.summary.scalar(k.split('/')[-1], eval_metric_ops[k][1])

        tensors_to_log = {'learning_rate': learning_rate}
        tf.summary.scalar('learning_rate', learning_rate)

        train_hooks = hooks_helper.get_train_hooks(['LoggingTensorHook'], model_dir=params['model_dir'], tensors_to_log=tensors_to_log, batch_size=params['batch_size'], use_tpu=params["use_tpu"])
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=logits, training_hooks=[logging_hook]+train_hooks, eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, predictions=logits, eval_metric_ops=eval_metric_ops)
    else:

        summary_hook = tf.train.SummarySaverHook(
            save_secs=1000,
            output_dir='./output/ckpt/pred',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

        # summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir='./output/ckpt/pred', summary_op=image_summary_op)
        return tf.estimator.EstimatorSpec(mode, predictions=logits, prediction_hooks=[summary_hook])
