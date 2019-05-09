import torch
import torch.nn as nn
import torch.nn.functional as F


class Online_Contrastive_Loss(nn.Module):
    def __init__(self, margin=2.0, num_classes=200):
        super(Online_Contrastive_Loss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, x, label):
        # compute pair wise distance
        n = x.size(0)
        xxt = torch.matmul(x, x.t())
        xn = torch.sum(torch.mul(x, x), keepdim=True, dim=-1)
        dist = xn.t() + xn - 2.0*xxt

        one_hot_label = torch.zeros(x.size(0), self.num_classes)
        one_hot_label.scatter_(1, label.unsqueeze(-1), 1)
        pmask = torch.matmul(one_hot_label, one_hot_label.t())
        nmask = (1-pmask)
        pmask[torch.eye(pmask.shape[0]) > 0] = 0.0

        pmask = pmask > 0
        nmask = nmask > 0

        ploss = torch.sum(torch.masked_select(dist, pmask)) /torch.sum(pmask)
        nloss = torch.sum(torch.clamp(self.margin - torch.masked_select(dist, nmask), min=0.0)) /torch.sum(nmask)
        #
        # mining the top k hardest negative pairs
        # neg_dist = torch.masked_select(-dist, nmask)
        # k = torch.sum(pmask)
        # neg_dist, _ = neg_dist.topk(k=k)
        # nloss = torch.sum(torch.clamp(self.margin + neg_dist, min=0.0))/k

        loss = (ploss + nloss)
        return loss



class Pooling_Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes, pooling_type='AVE'):
        super(Pooling_Classifier, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.pooling_type = pooling_type

        self.fc = nn.Linear(feat_dim, num_classes)
        # self.attention_layer = []
        if pooling_type == 'NAN':
            self.attention_layer = NAN_Attention(feat_dim)

    def forward(self, x, lst_lens):
        x = F.normalize(x)
        if self.pooling_type == 'AVE':
            lst_mx = []
            idx = 0
            for i in range(lst_lens.size(0)):
                set_x = x[idx:idx+lst_lens[i], :]
                idx += lst_lens[i]
                set_mx = set_x.mean(dim=0, keepdim=True)
                lst_mx.append(set_mx)

            lst_mx = torch.cat(lst_mx, dim=0)
            logits = self.fc(lst_mx)

            return lst_mx, logits

        else:
            lst_mx = []
            idx = 0
            for i in range(lst_lens.size(0)):
                set_x = x[idx:idx + lst_lens[i], :] # n*d
                idx += lst_lens[i]
                lst_mx.append(set_x.t().unsqueeze(0))

            lst_mx = torch.cat(lst_mx, dim=0)
            feats = self.attention_layer(lst_mx)

            logits = self.fc(feats)

            return feats, logits


class NAN_Attention(nn.Module):
    def __init__(self, feat_dim=128, set_size=20):
        super(NAN_Attention, self).__init__()
        self.q = nn.Parameter(torch.ones((1, 1, feat_dim)) * 0.0)
        # self.q = nn.Parameter(torch.Tensor((1, 1, feat_dim)))
        # nn.init.xavier_uniform_(self.q)
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.tanh = nn.Tanh()
        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()

    def forward(self, Xs):
        # Xs: N*C*K
        N, C, K = Xs.shape
        score = torch.matmul(self.q, Xs) # N*1*K
        score = F.softmax(score, dim=-1)

        r = torch.mul(Xs, score)
        r = torch.sum(r, dim=-1) # N*C
        new_q = self.fc(r) # N*C
        new_q = self.tanh(new_q)
        new_q = new_q.view(N, 1, C)

        new_score = torch.matmul(new_q, Xs)
        new_score = F.softmax(new_score, dim=-1)

        o = torch.mul(Xs, new_score)
        o = torch.sum(o, dim=-1) #N*C

        return o




