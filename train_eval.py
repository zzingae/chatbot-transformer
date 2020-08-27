import os
import argparse
import tensorflow as tf
import model
import numpy as np
import random
from dataloader import *
from sklearn.model_selection import train_test_split


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--save_checkpoints_steps', type=int, default=5000)
    parser.add_argument('--keep_checkpoint_max', type=int, default=10)

    parser.add_argument('--vocab_limit', type=int, default=5000)
    parser.add_argument('--max_length', type=int, default=25)
    # n Transformer encoder and n Transformer decoder
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    # multi-head splits into (hidden_size / num heads) and combines after multi-head attention
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=128)
    # inner layer dimension in the feedforward network (128->512->128)
    parser.add_argument('--filter_size', type=int, default=512)

    parser.add_argument('--data_path', type=str, default='./data/ChatBotData.csv')
    parser.add_argument('--pre_data_path', type=str, default='./data/pre_ChatBotData.csv')
    parser.add_argument('--vocab_path', type=str, default='vocabulary.txt')
    parser.add_argument('--model_dir', type=str, default='./output/ckpt')

    args = parser.parse_args()

    tf.reset_default_graph()
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    if args.num_gpus > 0:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
        config = tf.estimator.RunConfig(train_distribute=strategy,
                                        save_checkpoints_steps = args.save_checkpoints_steps, # save checkpoint and evaluate every N steps
                                        tf_random_seed=1,
                                        save_summary_steps = 100, # make summary every N steps
                                        keep_checkpoint_max=args.keep_checkpoint_max) # save checkpoint files up to N
    else:
        print('No GPU found. Using CPU!')
        config = tf.estimator.RunConfig()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    question, answer = load_data(args.data_path, args.pre_data_path)
    char2idx, idx2char, args.vocab_size = load_vocabulary(question+answer, os.path.join(args.model_dir,args.vocab_path), args.vocab_limit)

    # define estimator (vocab_size should be determined before)
    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
            model_dir=args.model_dir,
            params=vars(args),
            config=config
            )

    # split train and eval QnA
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=42)

    train_input = text2num(train_input, char2idx, args.max_length)
    train_label = text2num(train_label, char2idx, args.max_length)

    eval_input = text2num(eval_input, char2idx, args.max_length)
    eval_label = text2num(eval_label, char2idx, args.max_length)

    tf.logging.set_verbosity(tf.logging.INFO)

    # train up to args.max_steps unless args.max_epochs done
    train_spec = tf.estimator.TrainSpec(max_steps=args.max_steps,
                                      input_fn=lambda:input_fn(epoch=args.max_epochs, batch_size=args.batch_size, data=(train_input, train_label)))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(epoch=None, batch_size=args.batch_size, data=(eval_input, eval_label)))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
