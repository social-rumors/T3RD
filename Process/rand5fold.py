import random
from random import shuffle
import os
from sklearn.model_selection import StratifiedKFold


def loadkfoldData(data_path, obj, fold):
    skf = StratifiedKFold(n_splits=fold, shuffle=True)
    datasets = {}

    labelPath = os.path.join(data_path, "Weibo/Weibo_covid19_label_all.txt")
    print("loading weibo label:")
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[1], line.split('\t')[0]
        labelDic[eid] = label.lower()
    pairs = list(zip(labelDic.keys(), labelDic.values()))
    shuffle(pairs)
    keys, values = zip(*pairs)
    for n_fold, (train_index, test_index) in enumerate(skf.split(keys, values)):
        train = [keys[i] for i in train_index]
        test = [keys[i] for i in test_index]
        shuffle(train)
        shuffle(test)
        datasets[f'fold{n_fold}_train'] = test
        datasets[f'fold{n_fold}_test'] = train

    label_path_twitter = os.path.join(data_path, "Twitter/Twitter_label_all.txt")
    print("loading Twitter label")
    twitter_train = []
    labelDic = {}
    train_num = 0
    for line in open(label_path_twitter):
        line = line.rstrip()
        label, eid = line.split('\t')[1], line.split('\t')[0]
        labelDic[eid] = label.lower()
        twitter_train.append(eid)
        train_num += 1
    shuffle(twitter_train)
    datasets['high_resource'] = twitter_train

    return datasets
