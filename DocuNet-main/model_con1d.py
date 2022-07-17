import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from long_seq import process_long_input_new
from losses import balanced_loss as ATLoss
import torch.nn.functional as F
# from allennlp.allennlp.modules.matrix_attention import DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention
# from allennlp.modules.matrix_attention import DotProductMatrixAttention
# from allennlp.modules.matrix_attention import CosineMatrixAttention
# from allennlp.modules.matrix_attention import BilinearMatrixAttention
# from element_wise import ElementWiseMatrixAttention
from attn_unet import AttentionUNet


class MLP(nn.Module):
    def __init__(self, n_in, n_out, ):  # dropout=0
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class DocREModel1(nn.Module):
    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1, dropout=0.1):
        super().__init__()
        self.config = config
        self.bert_model = model
        self.mask_entatt = args.ues_entity_attention_mask  #是否屏蔽实体自身
        self.win_att = args.ues_window_attention  # 是否使用窗口0,1,2
        self.win_size = args.window_size
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.bertdrop = nn.Dropout(0.6)
        self.use_seg_net = args.use_seg_net  # 是否使用unet
        self.unet_in_dim = args.unet_in_dim  # 3
        self.unet_out_dim = args.unet_out_dim  # 256
        self.in_mlp = 512
        self.out_mlp = 256
        self.in_ent = emb_size * 2 + 20
        self.out_ent = 256
        self.in_cont = 768
        self.out_cont = 256
        # self.liner = nn.Linear(config.hidden_size, args.unet_in_dim)
        # self.liner = nn.Linear(config.hidden_size, args.unet_out_dim)
        self.min_height = args.max_height
        self.channel_type = args.channel_type
        self.dis2idx = torch.zeros(5000)  # , dtype='int64'    1-9-》0变19    1-12-》0变25
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis2idx[521:] = 10
        self.dis2idx[1024:] = 11
        self.dis2idx[2048:] = 12

        self.dis_embs = nn.Embedding(26, 20)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.ent_layer = MLP(self.in_ent, self.out_ent)
        self.att_layer = MLP(self.in_cont, self.out_cont)
        # self.rel_layer = nn.Conv2d(in_channels=emb_size * 2 + 20, out_channels=256, kernel_size=1, bias=False)
        # self.att_conv = nn.Sequential(
        #     nn.Dropout2d(0.2),
        #     nn.Conv2d(in_channels=768, out_channels=256, kernel_size=5, padding=2),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        # )
        self.dropout = nn.Dropout(dropout)

        if self.use_seg_net:
            self.out_mlp = args.unet_in_dim
            self.segmentation_net = AttentionUNet(input_channels=args.unet_in_dim,
                                                  class_number=args.unet_out_dim,
                                                  down_channel=args.down_dim)
        self.mlp = MLP(self.in_mlp, self.out_mlp)
        self.head_extractor = nn.Linear(1 * config.hidden_size + args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(1 * config.hidden_size + args.unet_out_dim, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        # self.pre_layer = nn.Linear(256, config.num_labels)

    def encode(self, input_ids, attention_mask,entity_pos):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input_new(self.bert_model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention


    def get_window_attention(self, win_flag, attention, batch, seq_length, start, end):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        win_size = self.win_size
        win_att = []
        # win_att.append(attention[batch, :, start + offset].unsqueeze(0))
        if win_flag == 1:  # 使用实体附近窗口机制
            # for w in range(win_size):
                # left_to_start = start + offset - w - 1
                # right_to_end = end + offset - 1 + w + 1
                # if left_to_start > 0:
                #     win_att.append(attention[batch, :, left_to_start].unsqueeze(0))
                # if right_to_end < seq_length - 1:
                #     win_att.append(attention[batch, :, right_to_end].unsqueeze(0))

            # new window
            left_to_start = start + offset - win_size - 1
            right_to_end = end + offset - 1 + win_size + 1
            if left_to_start < 0:
                left_to_start = 0
            if right_to_end > seq_length - 1:
                right_to_end = seq_length - 1
            att = attention[batch, :, start + offset].unsqueeze(0)
            mask = torch.zeros_like(att)
            mask[:, :, left_to_start:right_to_end] = 1
            att = torch.mul(att, mask)
            win_att.append(att)
            win_att = torch.cat(win_att, dim=0)
            win_att = torch.mean(win_att, dim=0, keepdim=True)
        elif win_flag == 2:  # 使用attention score排名机制
            att_score = attention[batch, :, start + offset].mean(1).squeeze(0)  # c维度的注意力分数
            _, top_pos = torch.topk(att_score, win_size)
            for pos in top_pos:
                if pos >= seq_length - 1:
                    continue
                win_att.append(attention[batch, :, pos + offset].unsqueeze(0))
            win_att = torch.cat(win_att, dim=0)
            win_att = torch.mean(win_att, dim=0, keepdim=True)
        return win_att.squeeze(0)  # (h, c)


    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        bs, h, _, c = attention.size()
        ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数
        mask_flag = self.mask_entatt
        win_flag = self.win_att
        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):  # 每个batch
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:  # 有多个mentions
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            if win_flag:
                                att = self.get_window_attention(win_flag, attention, i, c, start, end)
                            else:
                                att = attention[i, :, start + offset]  # (h, c)
                            if mask_flag:  # 计算实体注意力时去掉实体本身的字的权重
                                mask = torch.zeros_like(att)
                                for idx in range(c):
                                    if idx in range(start + offset + 1, end + offset):
                                        mask[:, idx] = 1
                                att = att.masked_fill(mask.eq(1), 0)

                            e_att.append(att)  # (head, c)

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        # e_att = attention[i, :, start + offset]
                        if win_flag:
                            e_att = self.get_window_attention(win_flag, attention, i, c, start, end)
                        else:
                            e_att = attention[i, :, start + offset]  # (h, c)
                        if mask_flag:  # 计算实体注意力时去掉实体本身的字的权重
                            mask = torch.zeros_like(e_att)
                            for idx in range(c):
                                if idx in range(start + offset + 1, end + offset):
                                    mask[:, idx] = 1
                            e_att = e_att.masked_fill(mask.eq(1), 0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            for _ in range(ne - entity_num-1):  # self.min_height
                entity_atts.append(e_att)  # 为啥多添加  类似于padding？  (42,12,244)=(最大实体数,注意力头,document词长度)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]


            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)

        new_entity_es = torch.zeros((bs, ne, 768), dtype=torch.float32)
        for j, x in enumerate(entity_es):
            new_entity_es[j, :x.shape[0], :x.shape[1]] = x
        return hss, tss, new_entity_es, entity_as

    def get_mask(self, ents, bs, ne, run_device):
        ent_mask = torch.zeros(bs, ne, device=run_device)
        rel_mask = torch.zeros(bs, ne, ne, device=run_device)
        for _b in range(bs):
            ent_mask[_b, :len(ents[_b])] = 1
            rel_mask[_b, :len(ents[_b]), :len(ents[_b])] = 1
        return ent_mask, rel_mask


    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i,h_index,t_index])
        htss = torch.stack(htss,dim=0)
        return htss

    def get_channel_map(self, sequence_output, entity_as):
        # sequence_output = sequence_output.to('cpu')
        # attention = attention.to('cpu')
        bs,_,d = sequence_output.size()
        ne = max([len(x) for x in entity_as])  # 本次bs中的最大实体数
        # ne = self.min_height

        index_pair = []
        for i in range(ne):
            tmp = torch.cat((torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)), dim=-1)
            index_pair.append(tmp)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)  # 从(ne,ne,2)=>(ne*ne,2)
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])  # 头实体的每个头的注意力(ne*ne, h, c)
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])  # 尾实体的每个头的注意力
            ht_att = (h_att * t_att).mean(1)  # 在注意力头上求平均 (ne*ne, c)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  # softmax功能
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)  # (ne*ne, 768) = (ne*ne, c) * (c, 768)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def get_dis_inputs(self, entity_pos):  # entity_pos=[[(start, end)],[(start, end),(start, end),(start, end)]]
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数
        bs = len(entity_pos)
        dis_inputs = []
        _dis_inputs = torch.zeros((ne, ne), dtype=torch.long)
        for b in range(bs):  # 每个batch
            for i, e in enumerate(entity_pos[b]):
                pos = e[0][0]
                _dis_inputs[i, :] += pos + offset  # 先出现的词到后出现的词的距离是 pos先-pos后
                _dis_inputs[:, i] -= pos + offset  # 加offside
            for i in range(ne):
                for j in range(ne):
                    if _dis_inputs[i, j] < 0:
                        _dis_inputs[i, j] = self.dis2idx[-_dis_inputs[i, j]] + 12  # 9
                    else:
                        _dis_inputs[i, j] = self.dis2idx[_dis_inputs[i, j]]
            _dis_inputs[_dis_inputs == 0] = 25  #  19
            dis_inputs.append(_dis_inputs.unsqueeze(0))
        dis_inputs = torch.cat(dis_inputs, dim=0)
        return dis_inputs

    def get_ent_map(self, x, y, dis_emb):  # x,y:(bs, ne, 768)  dis_emb:(bs, ne, ne, 20)
        M = x.shape[1]
        N = y.shape[1]
        fea_map = torch.cat([x.unsqueeze(2).repeat_interleave(N, 2), y.unsqueeze(1).repeat_interleave(M, 1)],
                            -1).to(dis_emb.device)  # .permute(0, 3, 1, 2).contiguous()
        if dis_emb is not None:
            fea_map = torch.cat([fea_map, dis_emb], -1)  # .permute(0, 3, 1, 2).contiguous()
        rel_map = self.activation(self.ent_layer(fea_map))  # 2*768 + 20 => 256
        return rel_map#.permute(0, 2, 3, 1).contiguous()


    def forward(self,
                input_ids=None,
                attention_mask=None,  # 有word的部分mask=1，否则补0
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask,entity_pos)

        bs, sequen_len, d = sequence_output.shape
        run_device = sequence_output.device.index
        ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数
        ent_mask, rel_mask = self.get_mask(entity_pos, bs, ne, run_device)  # 好像没什么用

        # get hs, ts and entity_embs >> entity_rs
        hs, ts, entity_embs, entity_as = self.get_hrt(sequence_output, attention, entity_pos, hts)
        dis_inputs = self.get_dis_inputs(entity_pos)   # (bs, ne, ne）
        dis_emb = self.dis_embs(dis_inputs.to(input_ids.device))  # (bs, ne, ne, 20)
        feature_ent = self.get_ent_map(entity_embs, entity_embs, dis_emb)  # feature_ent(bs,256,ne,ne)
        feature_cont = self.get_channel_map(sequence_output, entity_as)  # feature_cont(bs,ne,ne,768)
        feature_cont = self.att_layer(feature_cont)    # .permute(0, 3, 1, 2) feature_cont(bs,ne,ne,768) => feature_cont(bs,256,ne,ne)

        bs, ne, _, _ = feature_cont.size()
        feature_merge = torch.cat((feature_cont, feature_ent), dim=-1)#.permute(0, 2, 3, 1)  # (bs, ne, ne, 256)拼接=>(bs, ne, ne, 512)
        attn_map = self.mlp(feature_merge)# .permute(0, 2, 3, 1)  # (bs, ne, ne, 512)=>(bs, ne, ne, 256或3)
        if self.use_seg_net:
            attn_map = self.segmentation_net(attn_map.permute(0, 3, 1, 2))  # (bs, ne, ne, 3)=>(bs, ne, ne, 256)
        h_t = self.get_ht(attn_map, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)  # reshape
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)  # (htnum, 12, 64)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)  # (htnum, 12*64*64)
        logits = self.bilinear(bl)


        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels))
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), output)
        return output

