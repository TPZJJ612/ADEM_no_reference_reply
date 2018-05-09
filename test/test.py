import tensorflow as tf
import os
import re
import jieba
import xlrd

def cut_word(sentence):
    stop_list = []
    # for i in open("stopword.txt", "r"):
    with open("stopword.txt", mode='r', encoding='utf-8') as f:
        for i in f:
            stop_list.append(i.strip("\n"))
    res_list = []
    tmp_list = list(jieba.cut(sentence))
    for i in tmp_list:
        # 使用停用词表
        if i not in stop_list:
            res_list.append(i)
        # 不使用停用词表
        # res_list.append(i)
    return res_list

def preprocess_xlsx_label_data(xlsx_file):
    """
    数据预处理
    :param xlsx_file:
    :return:
    """
    # data = xlrd.open_workbook('level_zhuangbei_data.xlsx')
    data = xlrd.open_workbook(xlsx_file)

    table_test = data.sheets()[0]
    ncols_all = table_test.nrows
    data_list = []
    label_list = []

    for i in range(0, ncols_all):
        ask = table_test.row_values(i)[0]
        answer = table_test.row_values(i)[1]
        # 简单粗暴的把问答数据相加，后续待改进：
        tmp_data = ask + "*" + answer
        tmp_label = table_test.row_values(i)[2]
        tmp_data = cut_word(tmp_data)
        # print(tmp_data)
        data_list.append(tmp_data)
        label_list.append(tmp_label)

    with open("test.txt", mode='w+', encoding='utf-8') as fo:
        for i in range(len(data_list)):
            tmp_str = ""
            for j in range(len(data_list[i])):
                tmp_str = tmp_str + " " + data_list[i][j]
            tmp_str = tmp_str + "\t" + str(label_list[i])
            tmp_str = tmp_str.strip()
            fo.write(tmp_str)
            fo.write("\n")
    fo.close()




if __name__ == '__main__':
    input_path = "./data/test1000.xlsx"
    preprocess_xlsx_label_data(input_path)

