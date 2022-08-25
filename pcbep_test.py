from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, PointConv, radius, global_mean_pool
from sklearn import metrics
from PPFConv import PPFConv
from PointTransformerConv import PointTransformerConv
import sys
import math
import scipy.spatial
from torch_scatter import scatter_add
from optparse import OptionParser

Center = T.Center()
Normalscale = T.NormalizeScale()
Delaunay = T.Delaunay()
Normal = T.GenerateMeshNormals()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_point_pos(pos):
    # pos_AB=torch.cat([pos_A, pos_B])
    pos = pos - pos.mean(dim=-2, keepdim=True)
    # pos_B=pos_B-pos_AB.mean(dim=-2, keepdim=True)
    scale = (1 / pos.abs().max()) * 0.999999
    pos = pos * scale
    # scale_B = (1 / pos_B.abs().max()) * 0.999999
    # pos_B = pos_B * scale_B
    return pos


def load_data(data_path):
    print('loading data')
    data_list = []

    with open(data_path, 'r') as f:
        n_g = int(f.readline().strip())
        num = 0
        for i in range(n_g):  # for each protein
            n = int(f.readline().strip())  # atom number
            point_tag = []
            point_fea_pssm = []
            point_pos = []
            point_aa = []
            aa_y = []
            mask = []

            for j in range(n):
                row = f.readline().strip().split()
                point_tag.append(int(row[1]))
                mask.append(int(row[2]))
                pos, fea_pssm = np.array([float(w) for w in row[3:6]]), np.array([float(w) for w in row[6:]])
                point_pos.append(pos)
                point_fea_pssm.append(fea_pssm)
                point_aa.append(int(row[0]))

            flag = -1
            for i in range(len(point_aa)):
                if (flag != point_aa[i]):
                    flag = point_aa[i]
                    aa_y.append(point_tag[i])
            # print(aa_y)
            x = torch.tensor(point_fea_pssm, dtype=torch.float)  # 39
            y = torch.tensor(point_tag)
            pos = torch.tensor(point_pos, dtype=torch.float)  # 3
            mask = torch.tensor(mask)

            # pos=normalize_point_pos(pos)
            data = Data(x=x, y=y, pos=pos)
            # print(data.norm)

            for i in range(len(point_aa)):
                point_aa[i] = point_aa[i] + num
            num = num + len(aa_y)

            aa = torch.tensor(point_aa)
            # print(aa)
            number = len(aa_y)
            aa_y = torch.tensor(aa_y)

            data.aa = aa
            data.aa_y = aa_y
            data.num = number
            data.mask = mask

            data = Center(data)
            # data = Normalscale(data)
            data = Delaunay(data)
            data = Normal(data)

            data = data.to(device)
            data_list.append(data)
    # print(data_list)
    return data_list


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU(), Dropout(0.3))
        for i in range(1, len(channels))
    ])


def generate_normal(pos, batch):
    data_norm = []
    batch_list = torch.unique(batch)
    for b in batch_list:
        pos_temp = pos[batch == b]
        pos_temp = pos_temp - pos_temp.mean(dim=-2, keepdim=True)
        pos_temp = pos_temp.cpu().numpy()
        tri = scipy.spatial.Delaunay(pos_temp, qhull_options='QJ')
        face = torch.from_numpy(tri.simplices)

        data_face = face.t().contiguous().to(device, torch.long)
        pos_temp = torch.tensor(pos_temp).to(device)

        vec1 = pos_temp[data_face[1]] - pos_temp[data_face[0]]
        vec2 = pos_temp[data_face[2]] - pos_temp[data_face[0]]
        face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

        idx = torch.cat([data_face[0], data_face[1], data_face[2]], dim=0)
        face_norm = face_norm.repeat(3, 1)

        norm = scatter_add(face_norm, idx, dim=0, dim_size=pos_temp.size(0))
        norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]

        data_norm.append(norm)

    return torch.cat(data_norm, dim=0)


class PointTransformerConv1(torch.nn.Module):
    def __init__(self, r, in_channels, out_channels):
        super(PointTransformerConv1, self).__init__()
        self.k = None
        self.r = r
        self.pos_nn = MLP([6, out_channels])

        self.attn_nn = MLP([out_channels, out_channels])

        self.conv = PointTransformerConv(in_channels, out_channels,
                                         pos_nn=self.pos_nn,
                                         attn_nn=self.attn_nn)

    def forward(self, x, pos, normal, batch):
        # row, col = knn(pos, pos, self.k, batch, batch)
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, edge_index, normal, self.r)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.conv1 = PointTransformerConv1(5, in_channels=39 + 20, out_channels=128)
        self.conv2 = PointTransformerConv1(8.5, in_channels=39 + 20, out_channels=128)
        self.conv3 = PointTransformerConv1(10, in_channels=39 + 20, out_channels=128)
        self.neck = Seq(Lin(128 + 128 + 128, 512), BN(512), ReLU(), Dropout(0.3))
        self.conv4 = PointTransformerConv1(15, in_channels=512, out_channels=512)
        self.mlp = Seq(Lin(512, 256), BN(256), ReLU(), Dropout(0.3), Lin(256, out_channels))

    def forward(self, data):
        x0, pos, batch, normal, pool_batch, aa_num, mask = data.x, data.pos, data.batch, data.norm, data.aa, data.num, data.mask

        # atom to residue
        flag = torch.Tensor([-1]).to(device)
        num = -1
        for i in range(len(pool_batch)):
            if not torch.eq(pool_batch[i], flag):
                flag = pool_batch[i].clone()
                num = num + 1
                pool_batch[i] = torch.Tensor([num]).to(device)
            else:
                pool_batch[i] = torch.Tensor([num]).to(device)

        x1 = self.conv1(x0, pos, normal, batch)
        x2 = self.conv2(x0, pos, normal, batch)
        x3 = self.conv3(x0, pos, normal, batch)
        # print(batch)
        out = self.neck(torch.cat([x1, x2, x3], dim=1))
        out = global_max_pool(out, pool_batch)  # out-512
        # print(out)

        # residual batch
        # print(num)
        num_total = 0
        for i in range(len(aa_num)):
            num_total += aa_num[i]
        # print(num_total)
        aa_batch = torch.zeros(num_total).to(device)
        number = 0
        for m in range(len(aa_num)):
            # print(m)
            for n in range(aa_num[m].item()):
                aa_batch[n + number] = m
            number += aa_num[m].item()
        aa_batch = aa_batch.long()

        aa_pos = global_mean_pool(pos, pool_batch)

        aa_norm = generate_normal(aa_pos, aa_batch).to(device)

        out = self.conv4(out, aa_pos, aa_norm, aa_batch)
        out = self.mlp(out)

        mask = global_max_pool(mask, pool_batch)
        mask = mask == 1
        data.label = data.aa_y[mask]
        out = out[mask]
        # data.label = data.aa_y
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        pos_weight = torch.FloatTensor([1.0]).to(device)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def start(options):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_data(options.dataset)
    print(len(dataset))
    testloader = DataLoader(dataset, batch_size=1)


    checkpoint = options.checkpoint

    pred_total = []
    aa_total = []
    out_total = []

    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            out = model(data)
            out = F.sigmoid(out)
            out_total.extend(out.cpu().tolist())
            pred = out.ge(0.65).float()
            pred_total.extend(pred.detach().cpu().numpy())
            aa_total.extend(data.label.detach().cpu().numpy())

    pred_total = torch.tensor(pred_total)
    out_total = torch.tensor(out_total)
    pred_total = pred_total.squeeze()
    out_total = out_total.squeeze()

    aa_total = torch.tensor(aa_total)

    correct = int(pred_total.eq(aa_total).sum().item())
    tn, fp, fn, tp = confusion_matrix(aa_total, pred_total).ravel()
    print('tn' + str(tn) + 'tp' + str(tp) + 'fn' + str(fn) + 'fp' + str(fp))
    recall = tp / (tp + fn)
    print('recall:' + str(recall))
    sp = tn / (fp + tn)
    print('sp:' + str(sp))
    precision = tp / (tp + fp)
    print('precision:' + str(precision))
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)
    print('mcc:' + str(mcc))
    auc = metrics.roc_auc_score(aa_total, out_total)
    print('AUC:' + str(auc))
    ap = metrics.average_precision_score(aa_total, out_total)
    print('AP:' + str(ap))
    # out_total1 = np.rint(out_total)
    f1 = metrics.f1_score(aa_total, pred_total)
    print('f1:' + str(f1))

    out_total = out_total.tolist()
    aa_total = aa_total.tolist()
    with open(options.output, 'w') as f:
        for i in range(len(out_total)):
            f.write(str(aa_total[i]) + '\t' + str(out_total[i]) + '\n')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", type="string", dest="dataset", default='Data/data_feature_surface.txt',
                      help="Data file path entered.")
    parser.add_option("-c", "--checkpoint", type="string", dest="checkpoint", default='checkpoint.pt',
                      help="Checkpoint file path entered.")
    parser.add_option("-o", "--output", type="string", dest="output", default="result/result_pcbep.txt",
                      help="Output the results to a file.")
    (options, args) = parser.parse_args()
    start(options)
