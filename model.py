from official.transformer.model.model_params import BASE_PARAMS as Transformer_params
import official.transformer.utils.metrics as metrics
from official.utils.logs import hooks_helper
from official.transformer.model.transformer import Transformer
from transformer import MaskTransformer
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
    # 기존 dict 의 key,value 에 사용자 입력 값을 추가함
    # extend dict values to defaultdict
    _params = Transformer_params.copy()
    for k in params:
        v = params[k]
        _params[k] = v
    params = _params

    if mode == tf.estimator.ModeKeys.PREDICT: features['answer']=None

    # define transformer
    transformer = MaskTransformer(params, (mode == tf.estimator.ModeKeys.TRAIN))
    logits = transformer(features['question'], features['answer'])

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        # 네트워크 출력 logits 와 실제 answer 간의 loss 를 계산
        xentropy, weights = metrics.padded_cross_entropy_loss(logits, features['answer'], params["label_smoothing"], params["vocab_size"])
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        # loss 를 minimize
        learning_rate = get_learning_rate(params['learning_rate'], params['hidden_size'], params['learning_rate_warmup_steps'])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=params['optimizer_adam_beta1'], beta2=params['optimizer_adam_beta2'], epsilon=params['optimizer_adam_epsilon'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # 매 100번 마다 logitmax 과 answer 값을 보여줌
        logging_hook = tf.train.LoggingTensorHook({"logitmax": tf.argmax(logits[0], -1), "answer": features['answer'][0]}, every_n_iter=100)

        # 여러가지 metric 을 계산하여 보여줌 (accuracy, BLEU score, ..)
        eval_metric_ops = metrics.get_eval_metrics(logits, features['answer'], params)
        tensors_to_log = {}
        for k in eval_metric_ops:
            tensors_to_log[k.split('/')[-1]] = eval_metric_ops[k][1].name
            tf.summary.scalar(k.split('/')[-1], eval_metric_ops[k][1])

        tensors_to_log = {'learning_rate': learning_rate}
        tf.summary.scalar('learning_rate', learning_rate)

        train_hooks = hooks_helper.get_train_hooks(['LoggingTensorHook'], model_dir=params['model_dir'], tensors_to_log=tensors_to_log, batch_size=params['batch_size'], use_tpu=params["use_tpu"])
        # train
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=logits, training_hooks=[logging_hook]+train_hooks, eval_metric_ops=eval_metric_ops)
        # evaluate
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, predictions=logits, eval_metric_ops=eval_metric_ops)
    # predict
    else:
        # predict 시에도 summary 저장
        summary_hook = tf.train.SummarySaverHook(
            save_secs=1000,
            output_dir='./output/ckpt/pred',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

        return tf.estimator.EstimatorSpec(mode, predictions=logits, prediction_hooks=[summary_hook])
