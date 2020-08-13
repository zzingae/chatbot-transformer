import tensorflow as tf
import enum
import os
import re
from tqdm import tqdm
import pandas as pd


# FILTERS = "([~.,!?\"':;)(])"
FILTERS = "([~.,\"':;)(])"
PAD = "<PADDING>"
# STD = "<START>"
END = "<END>"
# UNK = "<UNKNWON>"

# MARKER = [PAD, STD, END, UNK]
MARKER = [PAD, END]
CHANGE_FILTER = re.compile(FILTERS)


def load_data(path, pre_path):

    if os.path.exists(pre_path):
        data_df = pd.read_csv(pre_path, header=0, encoding='utf-8')
        question, answer = list(data_df['Q']), list(data_df['A'])       
    else:
        from konlpy.tag import Twitter
        data_df = pd.read_csv(path, header=0, encoding='utf-8')
        question, answer = list(data_df['Q']), list(data_df['A'])
        
        twitter = Twitter()
        question = [cleaning_sentence(twitter, seq) for seq in tqdm(question)]
        answer = [cleaning_sentence(twitter, seq) for seq in tqdm(answer)]

        QnA = {'Q':question, 'A':answer}
        df = pd.DataFrame(QnA, columns=['Q', 'A'])
        df.to_csv(pre_path, index = False, header=True)

    return question, answer


def cleaning_sentence(morpher, sentence):
    morph_seq = " ".join(morpher.morphs(sentence))
    return re.sub(CHANGE_FILTER, "", morph_seq)
    

def load_vocabulary(sentences, path, vocab_limit=5000):

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as vocab_file:
            vocab_list = [line.strip() for line in vocab_file]
    else:
        words = [word for seq in sentences for word in seq.split()]
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word]+=1
            else:
                word_freq[word]=1

        vocab=sorted(word_freq, key=lambda k : word_freq[k], reverse=True)
        vocab_list = MARKER + vocab[:vocab_limit]

        with open(path, 'w') as vocabulary_file:
            for word in vocab_list:
                vocabulary_file.write(word + '\n')

    char2idx = {char: idx for idx, char in enumerate(vocab_list)}
    idx2char = {idx: char for idx, char in enumerate(vocab_list)}
    
    return char2idx, idx2char, len(char2idx)


def text2num(sentences, dictionary, max_length):

    sequences_input_index = []

    for sequence in sentences:
        sequence_index = []

        for word in sequence.split():
            if word in dictionary:
                sequence_index.append(dictionary[word])
            else:
                sequence_index.append(dictionary[PAD])

        if len(sequence_index) > max_length-1:
            sequence_index = sequence_index[:max_length-1]

        sequence_index += [dictionary[END]]
        sequence_index += [dictionary[PAD]] * (max_length - len(sequence_index))

        sequences_input_index.append(sequence_index)

    return sequences_input_index


def num2text(indices, end_idx, dictionary):
    try:
        num = indices.index(end_idx)
    except ValueError:
        # when there's no end token, use all indices
        num=len(indices)
    words = [dictionary[indices[n]] for n in range(num)]
    return ' '.join(words)


def train_parse_function():
    def _parse_function(question, answer):
        # >  https://www.tensorflow.org/guide/datasets
        return {"question": question, "answer" : tf.cast(answer, tf.int32)}
    return lambda a,b: _parse_function(a,b)


def pred_parse_function():
    def _parse_function(question):

        return {"question": question}
    return lambda a: _parse_function(a)


def input_fn(epoch, batch_size, data):
    # > reference : https://www.tensorflow.org/guide/datasets
    questions = tf.constant(data[0])
    answers = tf.constant(data[1])
    ds = tf.data.Dataset.from_tensor_slices((questions, answers))
    ds = ds.map(train_parse_function(), num_parallel_calls=8)

    if epoch is None:
        ds = ds.repeat(1)
    else:
        SHUFFLE_SIZE = len(data[0])
        ds = ds.shuffle(SHUFFLE_SIZE).repeat(epoch)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def pred_input_fn(batch_size, questions):

    ds = tf.data.Dataset.from_tensor_slices((questions))
    ds = ds.map(pred_parse_function(), num_parallel_calls=8)
    ds = ds.repeat(1)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds