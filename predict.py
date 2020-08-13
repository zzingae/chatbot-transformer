from konlpy.tag import Twitter
import os
import argparse
from dataloader import *
import model
import tensorflow as tf


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)

    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--extra_decode_length', type=int, default=0)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--filter_size', type=int, default=512) # Inner layer dimension in the feedforward network (128->512->128)

    parser.add_argument('--vocab_path', type=str, default='./data/vocabulary.txt')
    parser.add_argument('--model_dir', type=str, default='./output/ckpt')
    parser.add_argument('--pred_checkpoint', type=str, default='model.ckpt-100000')

    # error when spaces exist between characters
    parser.add_argument('--question', type=str, default='안녕하세요')

    args = parser.parse_args()

    assert os.path.exists(args.model_dir), 'cannot find model ckpt!'
    assert os.path.exists(args.vocab_path), 'cannot find vocab file!'

    config = tf.estimator.RunConfig()

    char2idx, idx2char, args.vocab_size = load_vocabulary(None, args.vocab_path)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
            model_dir=args.model_dir,
            params=vars(args),
            config=config
            )

    args.question = '감사합니다'

    twitter = Twitter()
    morph_question = [cleaning_sentence(twitter, args.question)]
    num_question = text2num(morph_question,char2idx,args.max_length)

    preds=estimator.predict(input_fn=lambda:pred_input_fn(args.batch_size, num_question),
                              checkpoint_path=os.path.join(args.model_dir, args.pred_checkpoint))

    for pred in preds:
        indices = pred['outputs'].tolist()
        # 최초 END 토큰이 나오기 전까지만 사용
        answer = num2text(indices, char2idx['<END>'], idx2char)
        print('Q: {}'.format(morph_question))
        print('A: {}'.format(answer))
        