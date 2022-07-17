import os
import os.path
import json
from datetime import datetime

import numpy as np

rel2id = json.load(open('./meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}

finrel2id = json.load(open('./meta/fin_rel2id.json', 'r'))
finid2rel = {value: key for key, value in finrel2id.items()}

def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    print('h_idx, preds', len(h_idx), len(preds))
    # assert len(h_idx) == len(preds)


    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train.json"), truth_dir)
    # fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, "dev.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    #if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        #in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r


def get_errors(tmp, path):
    '''
    return error predictions
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train.json"), truth_dir)
    # fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, "dev.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, h_idx, t_idx)] = r

    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])  # 所有docs的预测的三元组集合
    titleset2 = set([])
    errors = []
    for x in submission_answer:
        error = []
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        if (title, h_idx, t_idx) in std:
            if r == std[(title, h_idx, t_idx)]:
                continue
            else:
                error.append(title)
                error.append(h_idx)
                error.append(t_idx)
                error.append(r)
                error.append(std[(title, h_idx, t_idx)])
                errors.append(error)  # [[标题，头，尾预测关系，真实关系]]
    print('submission_answers,errors', len(submission_answer), len(errors))
    return errors

def to_fin(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    print('h_idx, preds', len(h_idx), len(preds))
    # assert len(h_idx) == len(preds)


    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': finid2rel[p],
                    }
                )
    print('res', len(res))
    return res
def gen_fin_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name, encoding='utf-8'))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train
    fact_in_train = set([])
    ori_data = json.load(open(data_file_name, encoding='utf-8'))

    #  新数据集的改动：vertexSet[label['h']][0]['name']   加了[0],新数据加了abbr，一个实体可能有俩指称
    for data in ori_data:
        vertexSet = data['entities']
        for label in data['triples']:
            fact_in_train.add((vertexSet[label['h']][0]['name'], vertexSet[label['t']][0]['name'], label['r']))
    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def fin_evaluate(tmp, path):
    '''
        changed from the Docred official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train = gen_fin_train_facts(os.path.join(path, "train.json"), truth_dir)
    # for train
    truth = json.load(open(os.path.join(path, "dev.json"), encoding='utf-8'))
    # for test
    # truth = json.load(open(os.path.join(path, "test.json"), encoding='utf-8'))
    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['doc_index']
        titleset.add(title)

        vertexSet = x['entities']
        title2vectexSet[title] = vertexSet

        for label in x['triples']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']

            std[(title, r, h_idx, t_idx)] = set([])#set(label['evidence'])  #######
            # tot_evidences += len(label['evidence'])###########
    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
    print('tmp, submission_answer, tot_relations', len(tmp), len(submission_answer), tot_relations)
    correct_re = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0  # 并没有
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            # correct_evidence += len(stdevi & evi)
            in_train_annotated = False
            #  新数据集的改动：vertexSet[label['h']][0]['name']   加了[0],新数据加了abbr，一个实体可能有俩指称
            if (vertexSet[h_idx][0]['name'], vertexSet[t_idx][0]['name'], r) in fact_in_train:
                in_train_annotated = True

            if in_train_annotated:
                correct_in_train_annotated += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    # evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    # evi_r = 1.0 * correct_evidence / tot_evidences
    # if evi_p + evi_r == 0:
    #     evi_f1 = 0
    # else:
    #     evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, 0, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r

def fin_get_errors(tmp, path):
    '''
        return error predictions
        '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    truth = json.load(open(os.path.join(path, "dev.json"), encoding='utf-8'))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['doc_index']
        titleset.add(title)

        vertexSet = x['entities']
        title2vectexSet[title] = vertexSet

        for label in x['triples']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            if (title, h_idx, t_idx) not in std:
                std[(title, h_idx, t_idx)] = []
                std[(title, h_idx, t_idx)].append(r)
            else:
                std[(title, h_idx, t_idx)].append(r)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])  # 所有docs的预测的三元组集合
    print('tmp, submission_answer', len(tmp), len(submission_answer))
    titleset2 = set([])
    errors = []
    errots_dic = []
    narels = []
    rights = 0
    for x in submission_answer:
        error = []
        error_dic = {}
        narel = []
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        if (title, h_idx, t_idx) in std:
            if r in std[(title, h_idx, t_idx)]:
                rights += 1
            else:
                error.append(title)
                error.append(h_idx)
                error.append(t_idx)
                error.append(r)
                error.append(std[(title, h_idx, t_idx)])
                errors.append(error)  # {(标题，头，尾):[预测关系，真实关系]}
                error_dic["title_h_t"] = [title, h_idx, t_idx]
                error_dic["pre_relation"] = r
                error_dic["rel_relation"] = std[(title, h_idx, t_idx)]
                errots_dic.append(error_dic)
        else:
            narel.append(title)
            narel.append(h_idx)
            narel.append(t_idx)
            narel.append(r)
            narels.append(narel)  # {(标题，头，尾):[预测关系，真实关系]}
    print('submission_answers:rights , errors , narels',
          len(submission_answer), rights, len(errors), len(narels))
    dir = os.path.join("./logs/fin/", "errors-%s.txt" % datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    with open(dir, 'a+') as f_log:
        json.dump(errots_dic, f_log, ensure_ascii=False, indent=1)
    return errors