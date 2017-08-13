# coding=utf-8
"""
数据特征转换
2017-07-27 张洛阳 杭州
"""
from sklearn.datasets import load_svmlight_file


def loadPredictData(dataPath):
    """
    load data
    :param dataPath:
    :return:
    """
    X, y = load_svmlight_file(dataPath)
    return X.toarray(), y


def convertPredictDataFormate(inputFile,
                              outputFile,
                              uidFile):
    """
    convert "userid(\t)features" to "label(\t)features"
    label is '0'
    :param inputFile: userid->features libsvm data format
    :param outputFile: label->features libsvm data format label is '0'
    :param uidFile: each line is a userid and line index correspondence to inputFile
    :return:
    """
    source_file = open(inputFile, "r")
    target_file = open(outputFile, "w")
    uidFile = open(uidFile, "w")
    lines = source_file.readlines()
    lines = map(lambda line: line.strip(), lines)
    for line in lines:
        sublist = line.strip().split("\t")
        uidFile.write(sublist[0] + "\n")
        line = line.replace(sublist[0], "0", 1)
        target_file.write(line + "\n")
    source_file.close()
    target_file.close()
    uidFile.close()

def writeLabelProb(labels,
                   probs,
                   outputFile):
    output_file = open(outputFile, "w")
    for label, prob in zip(labels, probs):
        label = int(label)
        output_file.write("{}\t{}\n".format(label, prob[label]))
    output_file.close()