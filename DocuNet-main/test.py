import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,  # 是否返回中间每层的attention输出
        )
        sequence_output = output[0]  # last_hidden_state
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention
def test():
    sent1 = ["Paul", "E.", "Pfeifer", "(", "born", "October", "15", ",", "1942", ")", "is", "an", "American", "jurist", "."]
    sent2 = ["He", "served", "in", "both", "houses", "of", "the", "Ohio", "General", "Assembly", "as", "a",
             "member", "of", "the", "Ohio", "Republican", "party", "and", "was", "most", "recently", "an", "Associate",
             "Justice", "of", "the", "Supreme", "Court", "of", "Ohio", "."]
    sent3 = "2016年11月28日，沣盈印月通过证券交易所集中交易直接买入亚威股份无限售流通股份148,500股，占公司总股本的0.04%。" \
            "本次增持后，沣盈印月持有亚威股份共18,573,722股，占亚威股份总股本的5.00%"
    dev_docs = json.load(open("./dataset/fin/dev.json", encoding='utf-8'))
    for doc in dev_docs:
        if doc["doc_index"] == 29:
            temp_doc = doc
            break
    sents = temp_doc["sentences"]
    entities = temp_doc["entities"]
    triples = temp_doc["triples"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_type", default="bert", type=str)  # bert roberta
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)  # bert-base-cased  roberta-base xlm-roberta-base bert-base-chinese
    parser.add_argument("--max_seq_length", default=512, type=int,  # 原来 1024
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    # config.add_cross_attention=True
    config.cls_token_id = tokenizer.cls_token_id  # 0
    config.sep_token_id = tokenizer.sep_token_id  # 2
    config.transformer_type = args.transformer_type
    model = AutoModel.from_pretrained(  # 读取bert或robert模型
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    if config.transformer_type == "bert":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
    elif config.transformer_type == "roberta":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id, config.sep_token_id]

    # new_sents = ''
    # for i, sent in enumerate(sents):
    #     sentmap[i] = len(new_sents)
    #     new_sents += sent
    # sentmap[i+1] = len(new_sents)
    # print(len(new_sents))
    #
    new_sents = []
    sent_map = []  # 一个document内每个句子的第n个词的分词起始位置
    entity_start, entity_end = [], []
    mention_types = []
    for entity in entities:
        for mention in entity['pos']:
            sent_id = mention[0]
            entity_start.append((sent_id, mention[1]))
            entity_end.append((sent_id, mention[2] - 1))
            # mention_types.append(mention['type'])
    word_map = {}
    for i_s, sent in enumerate(sents):
        new_map = {}  # 一个句子第n个词的分词起始位置
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                t = entity_start.index((i_s, i_t))
                # mention_type = mention_types[t]
                # special_token_i = entity_type.index(mention_type)
                # special_token = [
                #     '[unused' + str(special_token_i) + ']']  # '[unused1]' '[unused3]'..... '[unused11]'
            if (i_s, i_t) in entity_end:
                t = entity_end.index((i_s, i_t))
                # mention_type = mention_types[t]
                # special_token_i = entity_type.index(mention_type) + 50
                # special_token = [
                #     '[unused' + str(special_token_i) + ']']  # '[unused51]' '[unused53]'..... '[unused61]'
            new_map[i_t] = len(new_sents)
            word_map[len(new_sents)] = token
            new_sents.extend(tokens_wordpiece)
        new_map[i_t + 1] = len(new_sents)
        sent_map.append(new_map)

    # sent4 = new_sents
    # sent = tokenizer.tokenize(sent4)
    input_ids = tokenizer.convert_tokens_to_ids(new_sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    attention_mask = torch.tensor([[1.0] * len(input_ids)])
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    # attention_mask = torch.tensor(attention_mask, dtype=torch.float).unsqueeze(0)
    sequence_output, _ = process_long_input(model, input_ids, attention_mask, [tokenizer.cls_token_id], [tokenizer.sep_token_id])

    # entities =
    # triples =
    # sentmap
    for entity in entities:
        if entity["index"] == 2:
            ent1 = entity
        if entity["index"] == 3:
            ent2 = entity
            break
    ent1_pos = ent1['pos']
    ent2_pos = ent2['pos']
    new_pos = []
    for pos in ent1_pos:
        new_pos.append([sent_map[pos[0]][pos[1]], sent_map[pos[0]][pos[2]]])
    ent1_pos = new_pos
    new_pos = []
    for pos in ent2_pos:
        new_pos.append([sent_map[pos[0]][pos[1]], sent_map[pos[0]][pos[2]]])
    ent2_pos = new_pos
    ent_embs = []
    ids = []
    tokens = []
    for pos in ent1_pos:
        ent_embs.append([])
        ids.append([])
        for span in range(pos[0] + 1, pos[1] + 1):
            ent_embs[-1].append(sequence_output[0][span])
            ids[-1].append(input_ids[0][span])
        ent_embs[-1] = torch.stack(ent_embs[-1], dim=0).mean(0)
        ids[-1] = torch.stack(ids[-1], dim=0)
        tokens.append(tokenizer.convert_ids_to_tokens(ids[-1]))
    ent_embs = torch.stack(ent_embs, dim=0)
    ids = torch.stack(ids, dim=0)
    # tokens = tokenizer.convert_ids_to_tokens(ids)
    print(tokens)
    # 锐合创投 指代
    coref_pos = [[169, 185], [190, 194], [312, 316], [464, 468], [660, 664], [732, 736]]
    coref_embs = []
    coref_ids = []
    coref_tokens = []
    for pos in coref_pos:
        coref_embs.append([])
        coref_ids.append([])
        for span in range(pos[0] + 1, pos[1] + 1):
            coref_embs[-1].append(sequence_output[0][span])
            coref_ids[-1].append(input_ids[0][span])
        coref_embs[-1] = torch.stack(coref_embs[-1], dim=0).mean(0)
        coref_ids[-1] = torch.stack(coref_ids[-1], dim=0)
        coref_tokens.append(tokenizer.convert_ids_to_tokens(coref_ids[-1]))
    coref_embs = torch.stack(coref_embs, dim=0)
    print(coref_tokens)

    # 公司
    company_pos = [[218, 220], [235, 237], [336, 338], [470, 472], [655, 657], [666, 668]]
    company_embs = []
    company_ids = []
    company_tokens = []
    for pos in company_pos:
        company_embs.append([])
        company_ids.append([])
        for span in range(pos[0] + 1, pos[1] + 1):
            company_embs[-1].append(sequence_output[0][span])
            company_ids[-1].append(input_ids[0][span])
        company_embs[-1] = torch.stack(company_embs[-1], dim=0).mean(0)
        company_ids[-1] = torch.stack(company_ids[-1], dim=0)
        company_tokens.append(tokenizer.convert_ids_to_tokens(company_ids[-1]))
    company_embs = torch.stack(company_embs, dim=0)
    print(company_tokens)


    for i in range(6):
        sim = torch.cosine_similarity(ent_embs[0], company_embs[i + 0], dim=-1)
        print("上海泛微实体和  公司  的相似度", sim)
    for i in range(5):
        sim = torch.cosine_similarity(coref_embs[0], coref_embs[i + 1], dim=-1)
        print("上海锐合创业投资中心（有限合伙）实体和  锐合创投  的相似度", sim)

    sim = torch.cosine_similarity(ent_embs[0], coref_embs[0], dim=-1)
    print("上海泛微实体 和  上海锐合创业投资中心（有限合伙）实体  的相似度", sim)

    # attention = output[-1][-1]
    # attention = attention.mean(1)
    # test_tensor = attention[0, 6, :]
    # test_numpy = test_tensor.detach().numpy()
    # _, top_pos = torch.topk(test_tensor, 10)
    # print(top_pos)
    # top_words = tokenizer.convert_ids_to_tokens(input_ids[0, top_pos])
    # print(top_words)
    # plt.bar(range(len(test_numpy)), test_numpy)
    # plt.show()

    # 169-185 上海锐合创业投资中心（有限合伙）
    # 190-193  312-315  464-467  660-663  732-735 锐合创投
    # 218-220 235-237 336-338 470-472 655-657 666-668公司
    pass


# test()

a = torch.randn(3, 3)
b = torch.tensor([1, 1, 0])
print(a)
print(b)
print(a * b)
# b = torch.tensor([[0, 0, 0],
#                   [0, 0, 1],
#                   [1, 1, 1]])
# print(b < a)
# print((b < a).int())
# print(11 // 10)
# print(11 % 10)
# print(a)
# a = a.triu(diagonal=0)
# print(a)
# a = a.tril(diagonal=3 - 1)
# print(a)


# import spacy
# sent1 = "我喜欢打篮球，今天天气很好，我去篮球场"
# sent2 = "I like play basketball, weather is fine."
# sent = sent1
# nlp = spacy.load("zh_core_web_trf")
# newsent = nlp(sent)
# for word in newsent:
#     print(word, ': ', str(list(word.children)))
# att_conv = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
# linear = nn.Linear(5, 2)
# a = torch.zeros((2, 768, 10, 10))
# print(a.size())
# b = att_conv(a)


# print(b.size())
# a = torch.tensor([[[2., 1., 0., 1., 3.],
#                  [4., 1., 0., 1., 3.],
#                  [0., 1., 0., 1., 3.]]])
# b = torch.tensor([4., 2., 0., 3., 10.])
#
#
# c = torch.randn(4, 12, 20)
# d = torch.mean(c, dim=0, keepdim=True)
# print(d.size())
# # n1 = torch.norm(a)
# # print(n1)
# # n2 = torch.norm(a, float('nuc'))
# # print(n2)
# # b = torch.pow(n1, 2)
# s = [60, 61, 62]
# with open('./test.txt', 'a+') as f_log:
#     f_log.write(str(s) + '\n')
# a = torch.randn(1, 1, 10)
# a[:, :, 0:3] = 0
# a[:, :, 7:] = 0
# print(a)

def read_fin(max_seq_length=2048):
    print(max_seq_length)
read_fin(100)
read_fin(3000)




