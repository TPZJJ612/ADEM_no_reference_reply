#! /usr/bin/env python  
# -*- coding:utf-8 -*-  
#====#====#====#====  
#author:davidtu  
#FileName: cutword_test.py python3切词组件
#Version:1.0.0  
#====#====#====#====
import xlrd
import jieba

def cut_word(sentence):
    stop_list = []
    # for i in open("stopword.txt", "r"):
    with open('stopword.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            stop_list.append(line.strip("\n"))
    res_list = []
    tmp_list = list(jieba.cut(sentence))
    for i in tmp_list:
        if i not in stop_list:
            res_list.append(i)
    return res_list

def get_preprocess_data(input_txt, output_dir):
    """

    :param input:
    :param output:
    :return:
    """
    res_list = []
    positive_data = output_dir + "\\right.txt"
    negative_data = output_dir + "\\wrong.txt"
    positive_testdata = output_dir + "\\test_wrong.txt"
    negative_testdata = output_dir + "\\test_right.txt"
    with open(input_txt, mode='r', encoding='utf-8') as f:
        for line in f:
            tmp_slice = cut_word(line.encode("utf-8"))
            res_list.append(tmp_slice)
    # 写入测试样本
    # fo1 = open("test1.txt", "w")
    with open('test1.txt', mode='w+', encoding='utf-8') as f:
        for i in res_list:
            for j in range(len(i) - 1):
                tmp_val = i[j]
                f.write(i[j])
                f.write(" ")
            f.write(i[len(i) - 1])
            # f.write("\n")
    f.close()


# data = xlrd.open_workbook('level_zhuangbei_data.xlsx')
res_list = []
# for line in open("skill.txt", 'r'):
#     tmp_slice = cut_word(line.strip("\n"))
#     res_list.append(tmp_slice)
with open('skill.txt', mode='r', encoding='utf-8') as f:
    for line in f:
        tmp_slice = cut_word(line.encode("utf-8"))
        res_list.append(tmp_slice)

# 写入测试样本
# fo1 = open("test1.txt", "w")
with open('test1.txt', mode='w+', encoding='utf-8') as f:
    for i in res_list:
        for j in range(len(i) - 1):
            tmp_val = i[j]
            f.write(i[j])
            f.write(" ")
        f.write(i[len(i) -1])
        # f.write("\n")
f.close()







