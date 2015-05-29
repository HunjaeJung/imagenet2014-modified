def get_synset_words():
    dic = {}
    with open('/shared/tiny-imagenet-200/label.txt') as f:
        for l in f:
            dic[l.split()[0]] = l.split()[1]
    return dic
