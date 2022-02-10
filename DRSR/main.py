# coding=utf-8
import time

import torch
import torch.nn as nn
import networkx as nx
import os
import pandas as pd
import argparse
from model import MGCAT, DisGAT, GCN, GAT, DisenGCN
import numpy as np
import torch_geometric
from tqdm import tqdm

from task import classification, clustering


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default='brazil-flights')
    args.add_argument('--channel', type=int, default=8)
    args.add_argument('--input-dim', type=int, default=32)
    args.add_argument('--hidden-dim', type=int, default=32)
    args.add_argument('--out-dim', type=int, default=16)
    args.add_argument('--enc-layer', type=int, default=2)
    args.add_argument('--dec-layer', type=int, default=3)
    args.add_argument('--model', type=str, default='MGCAN')
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--gamma', type=float, default=0.0)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--l2', type=float, default=0.0)
    args.add_argument('--loop', type=int, default=100)
    args.add_argument('--device', type=int, default=0)

    return args.parse_args()


def acc(out, label):
    return torch.sum(torch.argmax(out, dim=1) == torch.argmax(label, dim=1)) / out.shape[0]


def inducer(graph, node):
    nebs = list(nx.neighbors(graph, node))
    sub_nodes = nebs + [node]
    sub_g = nx.subgraph(graph, sub_nodes)
    out_counts = np.sum(list(map(lambda x: len(list(nx.neighbors(graph, x))), sub_nodes)))
    return sub_g, out_counts, nebs


def get_base_features(graph):
    sub_graph_container = [[]] * graph.number_of_nodes()
    base_features = []
    for node in tqdm(range(0, graph.number_of_nodes())):
        sub_g, overall_counts, nebs = inducer(graph, node)
        in_counts = len(nx.edges(sub_g))
        sub_graph_container[node] = nebs
        deg = nx.degree(sub_g, node)
        trans = nx.clustering(sub_g, node)

        base_features.append(
            [in_counts, overall_counts, float(in_counts) / float(overall_counts),
             float(overall_counts - in_counts) / float(overall_counts), deg, trans])
    return np.array(base_features)


def run(args):
    print(args)
    torch.cuda.set_device(args.device)
    graph = nx.read_edgelist('../dataset/clf/{}.edge'.format(args.dataset), nodetype=int)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    label = pd.read_csv('../dataset/clf/{}.lbl'.format(args.dataset), dtype=int, header=None, sep=' ')
    label = label.sort_values(by=0).values
    label_b = torch.from_numpy(label)
    label_b = torch.eye(label[:, 1].max() + 1).index_select(dim=0, index=label_b[:, 1])

    edge_index = torch.tensor(np.transpose(graph.to_directed().edges()), dtype=torch.long)
    edge_index = torch_geometric.utils.sort_edge_index(edge_index)

    refex = torch.from_numpy(pd.read_csv('../cache/features/{}_features.csv'.format(args.dataset)).values).float()
    motif=refex

    adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0).to_sparse()
    eye = torch.eye(adj.shape[0]).to_sparse()
    H = eye

    if args.model == 'MGCAN':
        H = torch.stack([H for _ in range(args.channel)])
        model = DisGAT(args.channel, H.shape[-1], args.hidden_dim, args.out_dim, adj.shape[1], refex.shape[1],
                       motif.shape[1], label_b.shape[1], args.enc_layer, args.dec_layer).cuda()
    elif args.model == 'GCN':
        model = GCN(args.channel, H.shape[-1], args.hidden_dim, args.out_dim, adj.shape[1], refex.shape[1],
                    motif.shape[1], label_b.shape[1], args.enc_layer, args.dec_layer).cuda()
    elif args.model == 'GAT':
        model = GAT(args.channel, H.shape[-1], args.hidden_dim, args.out_dim, adj.shape[1], refex.shape[1],
                    motif.shape[1], label_b.shape[1], args.enc_layer, args.dec_layer).cuda()
    elif args.model == 'DisenGCN':
        H = torch.stack([H for _ in range(args.channel)])
        model = DisenGCN(args.channel, H.shape[-1], args.hidden_dim, args.out_dim, adj.shape[1], refex.shape[1],
                         motif.shape[1], label_b.shape[1], args.enc_layer, args.dec_layer).cuda()
    data = torch_geometric.data.Data(edge_index=edge_index, H=H).cuda()

    print(data)
    print(data.num_nodes, data.num_edges, data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_func = nn.MSELoss()
    # loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    model.train()
    total_time = 0
    for i in range(args.epochs):
        optimizer.zero_grad()
        start = time.time()

        embed, alpha, out2, out3 = model(data)
        refex_loss = loss_func(out2, refex.cuda())
        loss = refex_loss

        if args.model == 'MGCAN':
            reg_loss = alpha.pow(2).sum(dim=1).mean()
            total_loss = loss - args.gamma * reg_loss
        else:
            total_loss = loss
            reg_loss = 0

        total_loss.backward()
        optimizer.step()

        # train_acc = acc(out[train], label_b[train].cuda())
        # val_acc = acc(out[val], label_b[val].cuda())
        # test_acc = acc(out[test], label_b[test].cuda())
        # print("Epoch: {}/{}, Time: {}, Train loss:{}, Reg loss: {}, Train acc: {}, Val acc: {}, Test acc: {}"
        #       .format(i + 1, args.epochs, time.time() - start, loss.item(), reg_loss.item(), train_acc, val_acc,
        #               test_acc))
        total_time += time.time() - start
        print("Epoch: {}/{}, Time: {}, ReFeX loss: {}, Reg loss: {}, total loss: {}"
              .format(i + 1, args.epochs, time.time() - start, refex_loss.item(), reg_loss,
                      total_loss.item()))

        # if (i + 1) % 50 == 0:
        #     classification(embed.cpu().data.numpy(), label, loop=args.loop)
    print(f"Embedding time: {total_time}")
    model.eval()
    embed, alpha, _, _ = model(data)

    if args.model == 'MGCAN':
        eval_dict = classification(embed.cpu().data.numpy(), label, loop=args.loop)

    save_path = f'../embed/{args.model}/clf'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    columns = ['id'] + ['x_' + str(i) for i in range(embed.shape[1])]
    ids = np.array(range(embed.shape[0])).reshape((-1, 1))
    embedding = np.concatenate([ids, embed.cpu().data.numpy()], axis=1)
    embedding = pd.DataFrame(embedding, columns=columns)
    embedding.to_csv(os.path.join(save_path, '{}.emb'.format(args.dataset)), index=False)
    #
    # if args.model == 'MGCAN':
    #     for k in range(args.channel):
    #         t = torch.cat([data.edge_index, alpha[k][k].unsqueeze(0)]).t()
    #         t = t.cpu().data.numpy()
    #         t = pd.DataFrame(t)
    #         t[0] = t[0].astype(int)
    #         t[1] = t[1].astype(int)
    #         t.to_csv('{}_{}.csv'.format(args.dataset, k), index=False, header=['source', 'target', 'alpha'])
    #
    # classification(refex, label, loop=args.loop)
    return eval_dict


if __name__ == '__main__':
    args = parse_args()
    run(args)

    # result = []
    # for k in [1, 2, 4, 8, 16, 32, 64]:
    #     args.out_dim = k
    #     eval_dict = {
    #         'acc': 0.0,
    #         'f1-micro': 0.0,
    #         'f1-macro': 0.0,
    #     }
    #     for _ in range(20):
    #         eval = run(args)
    #         for key in eval_dict.keys():
    #             eval_dict[key] += eval[key]
    #
    #     for key in eval_dict.keys():
    #         eval_dict[key] /= 20
    #     result.append(eval_dict)
    # df = pd.DataFrame(result)
    # df.to_csv(f'result/{args.dataset}-dimension.csv', index=False)
