import tensorflow as tf
import enum
import os
import re
from tqdm import tqdm
import pandas as pd
import random
from sklearn.model_selection import train_test_split


# FILTERS = "([~.,!?\"':;)(])"
FILTERS = "([~.,\"':;)(])"
PAD = "<PADDING>"
# STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

# MARKER = [PAD, STD, END, UNK]
MARKER = [PAD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)


def load_data(path, pre_path):

    if os.path.exists(pre_path):
        data_df = pd.read_csv(pre_path, header=0, encoding='utf-8')
        question, answer, label = list(data_df['Q']), list(data_df['A']), list(data_df['label'])  
    else:
        from konlpy.tag import Twitter
        data_df = pd.read_csv(path, header=0, encoding='utf-8')
        question, answer, label = list(data_df['Q']), list(data_df['A']), list(data_df['label'])
        
        twitter = Twitter()
        question = [cleaning_sentence(twitter, seq) for seq in tqdm(question)]
        answer = [cleaning_sentence(twitter, seq) for seq in tqdm(answer)]

        QnA = {'Q':question, 'A':answer, 'label':label}
        df = pd.DataFrame(QnA, columns=['Q', 'A', 'label'])
        df.to_csv(pre_path, index = False, header=True)

    return question, answer, label


def cleaning_sentence(morpher, sentence):
    morph_seq = " ".join(morpher.morphs(sentence))
    return re.sub(CHANGE_FILTER, "", morph_seq)
    

def load_vocabulary(sentences, path, emotion_num=3, vocab_limit=5000):

    if os.path.exists(path):
        # with open(path, 'r', encoding='utf-8') as vocab_file:
        with open(path, 'r') as vocab_file:
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
        EMOT = ['e'+str(num) for num in range(emotion_num)]
        vocab_list = MARKER + EMOT + vocab[:vocab_limit]

        with open(path, 'w') as vocabulary_file:
            for word in vocab_list:
                vocabulary_file.write(word + '\n')

    char2idx = {char: idx for idx, char in enumerate(vocab_list)}
    idx2char = {idx: char for idx, char in enumerate(vocab_list)}
    
    return char2idx, idx2char, len(char2idx)

def my_train_test_split(question, answer, label):

    QnA={}
    for i in range(len(question)):
        QnA[question[i]+answer[i]]=i
    
    train_Q, eval_Q, train_A, eval_A = train_test_split(question, answer, test_size=0.33, random_state=42)

    train_L=[]
    for i in range(len(train_Q)):
        train_L.append(label[QnA[train_Q[i]+train_A[i]]])

    eval_L=[]
    for i in range(len(eval_Q)):
        eval_L.append(label[QnA[eval_Q[i]+eval_A[i]]])

    train_data = {'question': train_Q, 'answer': train_A, 'label': train_L}
    eval_data = {'question': eval_Q, 'answer': eval_A, 'label': eval_L}

    return train_data, eval_data

def text2num(sentences, dictionary, max_length):

    sequences_input_index = []

    for sequence in sentences:
        sequence_index = []

        for word in sequence.split():
            if word in dictionary:
                sequence_index.append(dictionary[word])
            else:
                sequence_index.append(dictionary[UNK])

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


def label_masking(label):

    end_token = 1

    # tf.where: If both x and y are None, then this operation returns the coordinates of true elements of condition
    actual_len = tf.squeeze(tf.where(tf.equal(label, end_token)))
    actual_len = tf.cast(actual_len,tf.int32)

    mask_num = tf.cond(tf.less(actual_len,3),
                    true_fn = lambda: tf.constant([0]),
                    false_fn= lambda: tf.random.uniform(shape=[1], minval=0, maxval=tf.cast(actual_len/3,tf.int32)+1, dtype=tf.int32))

    indices = tf.reshape(tf.range(start=0, limit=actual_len, delta=1), [actual_len,1])
    indices = tf.random.shuffle(indices)[mask_num[0]:]
    indices = tf.sort(indices,axis=0)

    reduced_label = tf.gather_nd(label, indices)
    return tf.concat([reduced_label, [1], tf.zeros(tf.size(label)-tf.size(reduced_label)-1,tf.int32)],axis=0)


def train_parse_function():
    def _parse_function(question, answer):
        # >  https://www.tensorflow.org/guide/datasets

        mask_question = label_masking(question)
        mask_answer = label_masking(answer)

        return {"question": mask_question, "answer" : mask_answer}

    return lambda a,b: _parse_function(a,b)


def pred_parse_function():
    def _parse_function(question):

        return {"question": question}
    return lambda a: _parse_function(a)


def input_fn(epoch, batch_size, data):
    # > reference : https://www.tensorflow.org/guide/datasets
    questions = tf.constant(data['question'])
    answers = tf.constant(data['answer'])
    ds = tf.data.Dataset.from_tensor_slices((questions, answers))
    ds = ds.map(train_parse_function(), num_parallel_calls=8)

    if epoch is None:
        ds = ds.repeat(1)
    else:
        SHUFFLE_SIZE = len(data['question'])
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