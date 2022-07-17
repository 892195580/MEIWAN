import json
import os
import re
from collections import defaultdict

from matplotlib import pyplot as plt
from prettytable import PrettyTable

relinfo = json.load(open("./dataset/docred/rel_info.json"))
data_dir = "./dataset/docred/"
train_file = "train_annotated.json"
dev_file = "dev.json"

fin_relinfo = json.load(open("./dataset/fin/fin_rel2id.json", encoding='utf-8'))
# train_file = os.path.join(data_dir, train_file)
# dev_file = os.path.join(data_dir, dev_file)



def read_ana(file_in, logfile):
    if file_in != 'dev.json':
        data_type = "train"
    else:
        data_type = "dev"
    print("文件类型：{}".format(data_type))
    file_in = os.path.join(data_dir, file_in)

    rel_dic = defaultdict(dict)
    type2rel_dict = defaultdict(dict)
    total_ent_num = 0  # 总实体数
    total_mul_ent_num = 0  # 实体有多个mention
    mul_ent_with_same_name_num = 0  # 多个mention都相同
    idx = 0
    different_mention_dict = {}


    with open(file_in, "r") as fh:
        docs = json.load(fh)
        for doc in docs:
            vertexSet = doc['vertexSet']
            labels = doc['labels']
            for ent in vertexSet:
                total_ent_num += 1
                if len(ent) > 1:
                    total_mul_ent_num += 1
                    entset = set()
                    for men in ent:
                        entset.add(men["name"])
                    if len(entset) > 1:
                        different_mention_dict[str(idx)] = list(entset)
                        idx += 1
                    else:
                        mul_ent_with_same_name_num += 1
            for label in labels:
                rid = label['r']
                hid = label['h']
                tid = label['t']
                rname = relinfo[rid]
                htype = vertexSet[hid][0]['type']
                ttype = vertexSet[tid][0]['type']
                httype = str((htype, ttype))
                if rname in rel_dic:
                    if httype in rel_dic[rname]:
                        rel_dic[rname][httype] += 1
                    else:
                        rel_dic[rname][httype] = 1
                else:
                    rel_dic[rname][httype] = 1

                if httype in type2rel_dict:
                    if rname in type2rel_dict[httype]:
                        type2rel_dict[httype][rname] += 1
                    else:
                        type2rel_dict[httype]['relation_nums'] += 1
                        type2rel_dict[httype][rname] = 1
                else:
                    type2rel_dict[httype]['relation_nums'] = 1
                    type2rel_dict[httype][rname] = 1

    # logfile.write("\n{}\n".format(file_in))
    # json.dump(rel_dic, logfile, ensure_ascii=False, indent=1)
    print("总实体数，有多个mention实体数，多个mention都相同实体数：{}，{}，{}".format(total_ent_num, total_mul_ent_num, mul_ent_with_same_name_num))
    if data_type == "train":
        json.dump([different_mention_dict, type2rel_dict], open(os.path.join('./dataset/docred', 'train_ana.json'), mode='w', encoding='utf-8'), ensure_ascii=False, indent=1)
    else:
        json.dump([different_mention_dict, type2rel_dict], open(os.path.join('./dataset/docred', 'dev_ana.json'), mode='w', encoding='utf-8'), ensure_ascii=False, indent=1)

def findabbr(sents):
    abbr_list = []
    pattern = re.compile('（以下简称.+?”）')
    for idx, sent in enumerate(sents):
        targets = pattern.findall(sent)
        if targets != []:
            print(targets)
            for target in targets:
                pattern1 = re.compile('“.*”）')
                target1 = pattern1.findall(target)
                if target1 != []:
                    abbr = target1[0][1:-2]
                    print(abbr)
                    if len(abbr) > 2:
                        abbr_list.append(abbr)
    return abbr_list

def read_fin_ana(file_type, log_file):
    if file_type == 'train':
        file_path = './dataset/abbr_fin/train.json'
    elif file_type == 'dev':
        file_path = './dataset/abbr_fin/dev.json'
    elif file_type == 'test':
        file_path = './dataset/abbr_fin/test.json'
    print("文件类型：{}".format(file_type))
    sentence_dict = []  # document句子总长度
    lower_512 = 0
    between_512_and_1024 = 0
    between_1024_and_2048 = 0
    upper_2048 = 0

    total_sent_num = 0
    total_ent_num = 0
    total_tri_num = 0
    # ana_dict = {}
    # ana_dict['triples_nums'] = 0
    # ana_dict['intra_tri'] = 0
    # ana_dict['inter_tri'] = 0
    # file_ent_nums = 0
    # file_ent_pair_nums = 0
    # file_tri_nums = 0
    with open(file_path, "r", encoding='utf-8') as fh:
        docs = json.load(fh)
        for doc in docs:
            entities = doc['entities']
            triples = doc['triples']
            sents = doc['sentences']
            doc_len = 0
            for sent in sents:
                doc_len += len(sent)
            if doc_len <= 512:
                lower_512 += 1
            elif doc_len > 512 and doc_len <=  1024:
                between_512_and_1024 += 1
            elif doc_len > 1024 and doc_len <=  2048:
                between_1024_and_2048 += 1
            elif doc_len > 2048:
                upper_2048 += 1
            sentence_dict.append(doc_len)
            total_sent_num += doc_len
            total_ent_num += len(entities)
            total_tri_num += len(triples)
        print("len-doc:", len(docs))
        print("lower_512:", lower_512, "\nbetween_512_and_1024:", between_512_and_1024,
          "\nbetween_1024_and_2048", between_1024_and_2048, "\nupper_2048:", upper_2048)
        print("total_sent_num:", total_sent_num)
        print("total_ent_num:", total_ent_num)
        print("total_tri_num:", total_tri_num)
    # file_ent_nums += len(entities)
    # file_tri_nums += len(triples)
    # file_ent_pair_nums += len(entities) * (len(entities) - 1)
    # for triple in triples:
    #     flag_inter = 1
    #     ana_dict['triples_nums'] += 1
    #     hid = triple['h']
    #     tid = triple['t']
    #     for posh in entities[hid]['pos']:
    #         for post in entities[tid]['pos']:
    #             if posh[0] == post[0]:
    #                 flag_inter = 0
    #     if flag_inter == 1:
    #         ana_dict['inter_tri'] += 1
    #     else:
    #         ana_dict['intra_tri'] += 1


    # print("实体数：", file_ent_nums)
    # print("三元组数：", file_tri_nums)
    # print("实体对数：", file_ent_pair_nums)
    # if file_type == "train":
    #     json.dump(ana_dict, open(os.path.join('./dataset/fin', 'train_ana.json'), mode='a+'), ensure_ascii=False, indent=1)
    # else:
    #     json.dump(ana_dict, open(os.path.join('./dataset/fin', 'dev_ana.json'), mode='a+'), ensure_ascii=False, indent=1)

def process_errors():
    error_list = []

    with open("./dataset/fin/errors.json", encoding='utf-8') as f:
        line = f.readline()
        errors = line.split('],')
        for idx, error in enumerate(errors):
            err = []
            title_s = re.search(r'\'', error, flags=0).start()
            pos_comma = re.finditer(r',', error, flags=0)
            pos_comma = [pos.span()[0] for pos in pos_comma]
            # pos_quotation = re.finditer(r'\'', error, flags=0)
            # pos_quotation = [pos.span()[0] for pos in pos_quotation]
            # pos_rbracket = re.finditer(r'\)', error, flags=0)
            # pos_rbracket = [pos.span()[0] for pos in pos_rbracket]
            title = error[title_s+1:pos_comma[-4]-1]
            hid = error[pos_comma[-4] + 2: pos_comma[-3]]
            tid = error[pos_comma[-3] + 2: pos_comma[-2]]
            prer = error[pos_comma[-2] + 3: pos_comma[-1]-1]
            trur = error[pos_comma[-1] + 3: len(error)-1]
            err.append(title)
            err.append(str(hid))
            err.append(str(tid))
            err.append(prer)
            err.append(trur)
            error_list.append(err)
    with open("./dataset/fin/new_errors.json", mode='w', encoding='utf-8') as f:  # , encoding='utf-8'
        json.dump(error_list, f, ensure_ascii=False, indent=1)


def fin_process_errors():
    error_list = []

    with open("./dataset/fin/errors.json", encoding='utf-8') as f:
        line = f.readline()
        errors = line.split('],')
        for idx, error in enumerate(errors):
            err = []
            if idx == len(errors) - 1:
                error = error[:-2]
            pos_bracket = re.finditer(r'\[', error, flags=0)
            pos_bracket = [pos.span()[0] for pos in pos_bracket]
            pos_comma = re.finditer(r',', error, flags=0)
            pos_comma = [pos.span()[0] for pos in pos_comma]

            title = error[pos_bracket[-2] + 1:pos_comma[0]]
            hid = error[pos_comma[0] + 2: pos_comma[1]]
            tid = error[pos_comma[1] + 2: pos_comma[2]]
            prer = error[pos_comma[2] + 3: pos_comma[3] - 1]
            true = []
            truegroup = error[pos_comma[3] + 3: len(error) - 1]
            truegroup = truegroup.split(', ')
            true = [tru[1:-1] for tru in truegroup]

            err.append(title)
            err.append(str(hid))
            err.append(str(tid))
            err.append(prer)
            err.append(true)
            error_list.append(err)
    with open("./dataset/fin/new_errors.json", mode='w', encoding='utf-8') as f:  # , encoding='utf-8'
        json.dump(error_list, f, ensure_ascii=False, indent=1)

def fin_process_new_errors():
    with open("./dataset/fin/new_errors.json", encoding='utf-8') as f:
        errors = json.load(f)
        print('错误预测总数：', len(errors))
        error_dict = {}
        for error in errors:
            true_group = error[-1]
            pre = error[-2]
            for true in true_group:
                (t, p) = (fin_relinfo[2][true], fin_relinfo[2][pre])
                if (t, p) in error_dict:
                    error_dict[(t, p)] += 1
                else:
                    error_dict[(t, p)] = 1

        a = sorted(error_dict.items(), key=lambda x: x[1], reverse=True)
        print(a)
        x = PrettyTable(["类型", "数量"])
        x.padding_width = 2
        for a1 in a:
            x.add_row([a1[0], a1[1]])
        x.align["类型"] = 'l'
        print(x)

def docred_error_ana():
    with open("./dataset/docred/new_errors.json", encoding='utf-8') as f, \
            open("./dataset/docred/dev_ana.json")as f2, \
            open("./dataset/docred/dev.json")as dev:
        docs = json.load(dev)
        etype2rel = json.load(f2)
        errors = json.load(f)
        title_list = [doc['title'] for doc in docs]
        error_nums = len(errors)
        wrong_type2rel_range = 0
        for error in errors:
            title = error[0]
            hid = int(error[1])
            tid = int(error[2])
            prer = error[3]
            trur = error[4]
            prer = relinfo[prer]
            trur = relinfo[trur]
            if title in title_list:
                title_idx = title_list.index(title)
            if title_idx:
                htype = docs[title_idx]["vertexSet"][hid][0]["type"]
                ttype = docs[title_idx]["vertexSet"][tid][0]["type"]
                if str((htype, ttype)) in etype2rel:
                    if prer not in etype2rel[str((htype, ttype))]:
                        wrong_type2rel_range += 1
        print("error_nums, wrong_type2rel_range:", error_nums, wrong_type2rel_range)

def docred_sen2triples_ana():
    with open("./dataset/docred/dev.json")as dev:
        docs = json.load(dev)
        sen2triples = []
        for doc in docs:
            sents = doc['sents']
            entities = doc['vertexSet']
            triples = doc['labels']
            new_triples = {}
            ent2triples = {}
            for i, ent in enumerate(entities):
                for m in ent:
                    ent2triples[str((m['sent_id'], m['pos'][0]))] = i
            new_sen = []
            for sid, sen in enumerate(sents):
                new_sen.append('')
                for wid, word in enumerate(sen):
                    if str((sid, wid)) in ent2triples:
                        new_sen[-1] = new_sen[-1] + ' {' + str(ent2triples[str((sid, wid))]) + '}'
                    new_sen[-1] = new_sen[-1] + ' ' + word
            for i, triple in enumerate(triples):
                new_triples[str(i)] = (str(triple['h']) + ' :' + entities[triple['h']][0]['name'],
                                       str(triple['t']) + ' :' + entities[triple['t']][0]['name'],
                                       triple['r'] + ' :' + relinfo[triple['r']])
            sen2triples.append({})
            sen2triples[-1]['sent'] = new_sen
            sen2triples[-1]['triples'] = new_triples
        with open("./dataset/docred/sen2triples.json", mode='w', encoding='utf-8') as f:  # , encoding='utf-8'
            json.dump(sen2triples, f, ensure_ascii=False, indent=1)

def fin_sen2triples_ana():
    with open("./dataset/fin/test_dev_data.json", encoding='utf-8')as dev:
        docs = json.load(dev)
        sen2triples = []
        intertriples = []
        ana_dict = {}
        ana_dict['triples_nums'] = 0
        ana_dict['intra_tri'] = 0
        ana_dict['inter_tri'] = 0
        for doc in docs:
            sents = doc['sentences']
            entities = doc['entities']
            triples = doc['triples']
            new_triples = {}
            ent2triples = {}
            inter_triples = {}
            for i, ent in enumerate(entities):
                for m in ent['pos']:
                    ent2triples[str((m[0], m[1]))] = i
            new_sen = []
            for sid, sen in enumerate(sents):
                new_sen.append('')
                for wid, word in enumerate(sen):
                    if str((sid, wid)) in ent2triples:
                        new_sen[-1] = new_sen[-1] + ' {' + str(ent2triples[str((sid, wid))]) + '}'
                    new_sen[-1] = new_sen[-1] + word
            num_inter = 0
            for i, triple in enumerate(triples):
                new_triples[str(i)] = (str(triple['h']) + ' :' + entities[triple['h']]['name'],
                                       str(triple['t']) + ' :' + entities[triple['t']]['name'],
                                       triple['r'] + ' :' + fin_relinfo[2][triple['r']])
                flag_inter = 1

                hid = triple['h']
                tid = triple['t']
                for posh in entities[hid]['pos']:
                    for post in entities[tid]['pos']:
                        if posh[0] == post[0]:
                            flag_inter = 0
                if flag_inter == 1:
                    num_inter += 1
                    inter_triples[str(i)] = (str(triple['h']) + ' :' + entities[triple['h']]['name'],
                                             str(triple['t']) + ' :' + entities[triple['t']]['name'],
                                             triple['r'] + ' :' + fin_relinfo[2][triple['r']])
            if num_inter > 0:
                intertriples.append({})
                intertriples[-1]['sent'] = new_sen
                intertriples[-1]['triples'] = inter_triples

            sen2triples.append({})
            sen2triples[-1]['sent'] = new_sen
            sen2triples[-1]['triples'] = new_triples
        with open("./dataset/fin/fin_sen2triples.json", mode='w', encoding='utf-8') as f:  # , encoding='utf-8'
            json.dump(sen2triples, f, ensure_ascii=False, indent=1)
        with open("./dataset/fin/fin_intertriples.json", mode='w', encoding='utf-8') as f:  # , encoding='utf-8'
            json.dump(intertriples, f, ensure_ascii=False, indent=1)


def docred_ana():
    with open("./docred_ana.log", "a+") as log:
        read_ana(train_file, log)
        read_ana(dev_file, log)

def fin_ana():
    with open("./fin_ana.log", "a+") as log:
        read_fin_ana('train', log)
        read_fin_ana('dev', log)
        read_fin_ana('test', log)

# docred_ana()
fin_ana()
# fin_process_errors()
# fin_process_new_errors()
# docred_error_ana()
# docred_sen2triples_ana()
# fin_sen2triples_ana()