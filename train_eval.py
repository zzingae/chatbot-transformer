import os
import argparse
import tensorflow as tf
import model
import numpy as np
import tarfile
import random

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2)
    parser.add_argument('--model_dir', type=str, default='./output/ckpt')
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--max_decode_length', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--num_units', type=int, default=512)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--dataset_dir', type=str, default="/data")
    parser.add_argument('--model', type=str, default="transformer")
    parser.add_argument('--eval_dir', type=str, default="/data")
    parser.add_argument('--max_steps', type=int, default=150*1000) # 150k
    parser.add_argument('--eval_num', type=int, default=40495)
    parser.add_argument('--save_checkpoints_steps', type=int, default=30000)
    parser.add_argument('--keep_checkpoint_max', type=int, default=10)
    parser.add_argument('--BNon', type=int, default=0)
    # removing tokens that occur less than N times in training data
    parser.add_argument('--vocab_freq_limit', type=int, default=200)
    parser.add_argument('--extract_tarfile', type=str, default="/data")

    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--filter_size', type=int, default=2048) # Inner layer dimension in the feedforward network.

    args = parser.parse_args()
    
    if args.extract_tarfile is not None:
        print("-BEGIN- Extracting all dataset")
        tar = tarfile.open(args.extract_tarfile)
        tar.extractall(os.path.split(args.extract_tarfile)[0])
        print("-DONE- Extracting all dataset")
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    tf.reset_default_graph()
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    if args.num_gpus > 0:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
        config = tf.estimator.RunConfig(train_distribute=strategy,
                                        save_checkpoints_steps = args.save_checkpoints_steps, # save checkpoint and evaluate using it for every N steps
                                        tf_random_seed=1, # for reproducing trained network
                                        save_summary_steps = 100, # make summary every N steps
                                        keep_checkpoint_max=args.keep_checkpoint_max) # save checkpoint files up to N
    else:
        print('No GPU found. Using CPU!')
        config = tf.estimator.RunConfig()
    
    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
            model_dir=args.model_dir,
            params=vars(args),
            config=config
            )

    if args.debug == True:
        tf.logging.set_verbosity(tf.logging.INFO)

    # https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
    if args.eval_dir is None:
        estimator.train(input_fn=lambda:model.input_fn(
                                        args.epochs, args.batch_size, dataset_dir=args.dataset_dir, max_length=args.max_decode_length, 
                                        image_size=(args.height, args.width), vocabulary=vocabulary), 
                                        max_steps=args.max_steps)
    else:
        train_spec = tf.estimator.TrainSpec(input_fn=lambda:model.input_fn(
                                            args.epochs, args.batch_size, dataset_dir=args.dataset_dir, max_length=args.max_decode_length, 
                                            image_size=(args.height, args.width), vocabulary=vocabulary), 
                                            max_steps=args.max_steps)

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda:model.input_fn(
                                            epochs=None, batch_size=args.batch_size, dataset_dir=args.eval_dir, max_length=args.max_decode_length, 
                                            image_size=(args.height, args.width), vocabulary=vocabulary))

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
