import numpy as np
import re
import itertools
from collections import Counter
import codecs

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(label_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    tmp_examples = list(open(label_data_file, "r", encoding='utf8').readlines())
    # print(positive_examples)
    tmp_xdata = [s.strip().split("\t")[0] for s in tmp_examples]
    tmp_label = [s.strip().split("\t")[1] for s in tmp_examples]
    y = []
    # x_text = positive_examples + negative_examples
    x_text = tmp_xdata
    for i in tmp_label:
        # if i == "1.0":
        #     y.append([1, 0, 0])
        # elif i == "2.0":
        #     y.append([0, 1, 0])
        # elif i == "3.0":
        #     y.append([0, 0, 1])
        y.append(i)

    return [x_text, y]

def load_data_without_labels(label_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    tmp_examples = list(open(label_data_file, "r", encoding='utf8').readlines())
    # print(positive_examples)
    tmp_xdata = [s.strip() for s in tmp_examples]
    x_text = tmp_xdata
    return x_text

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
