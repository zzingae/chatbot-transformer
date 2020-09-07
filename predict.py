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

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--extra_decode_length', type=int, default=0)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--filter_size', type=int, default=2048) # Inner layer dimension in the feedforward network (128->512->128)

    parser.add_argument('--vocab_path', type=str, default='vocabulary.txt')
    parser.add_argument('--model_dir', type=str, default='./output/ckpt-mask-10000')

    # error when spaces exist between characters
    parser.add_argument('--question', type=str, default='헤어지자')
    # https://github.com/songys/Chatbot_data
    # 일상다반서 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링
    parser.add_argument('--num_emotion', type=int, default=3)

    args = parser.parse_args()

    assert os.path.exists(args.model_dir), 'cannot find model ckpt!'
    assert os.path.exists(os.path.join(args.model_dir,args.vocab_path)), 'cannot find vocab file!'

    config = tf.estimator.RunConfig()

    char2idx, idx2char, args.vocab_size = load_vocabulary(None, os.path.join(args.model_dir,args.vocab_path))

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
            model_dir=args.model_dir,
            params=vars(args),
            config=config
            )

    # args.question = '감사합니다'

    twitter = Twitter()
    morph_question = [cleaning_sentence(twitter, args.question)]
    num_question = text2num(morph_question,char2idx,args.max_length-1)
    # add emotion token
    emot_questions=[]
    for e in range(args.num_emotion):
        emot_questions.append([char2idx['e'+str(e)]] + num_question[0])

    # latest checkpoint will be used for prediction unless specified
    preds=estimator.predict(input_fn=lambda:pred_input_fn(args.batch_size, emot_questions))
                            #   checkpoint_path=os.path.join(args.model_dir, 'model.ckpt-100000'))

    for pred in preds:
        indices = pred['outputs'].tolist()
        # 최초 END 토큰이 나오기 전까지만 사용
        print('Q: {}'.format(morph_question))
        for b in range(args.beam_size):
            answer = num2text(indices[b][1:], char2idx['<END>'], idx2char)
            print('top {}: {}'.format(b+1,answer))
        