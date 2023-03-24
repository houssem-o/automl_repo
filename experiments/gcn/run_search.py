import torch

torch.manual_seed(0)
import torch.nn as nn
import numpy as np

np.random.seed(0)
import pickle
from torch_geometric.nn import GCNConv, aggr

from src.ensemble_pyg import EnsemblePyG
from src.search_pyg import BenchSearchPyG
from utils.logs_utils import plot_average_logs_multiple_experiments

embedding_dim = 512

def get_pyg_data_list(input_data, target_data_lst=None):
    def transform_edge_index(v, ei):
        nei = torch.clone(ei)
        cols = set(range(10))
        if v[0] == 0.0:
            cols = cols - {0, 3, 4, 6, 7}
        if v[1] == 0.0:
            cols = cols - {1, 5}
        if v[2] == 0.0:
            cols = cols - {4, 6}
        if v[3] == 0.0:
            cols = cols - {2, 9}
        if v[4] == 0.0:
            cols = cols - {3, 7}
        if v[5] == 0.0:
            cols = cols - {1, 4, 5, 6, 8}
        return nei[:, list(cols)]

    # Constructing graphs
    #                           0  1  2  3  4  5  6  7  8  9
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 5, 6, 4],
                               [1, 2, 4, 5, 3, 6, 6, 7, 7, 7]], dtype=torch.long)
    # x_ = [input, op1, op2, op3, op4, op5, op6, output]
    data_list = []
    if target_data_lst is None:
        for v in input_data:
            vector = torch.Tensor([0.0] + v.tolist() + [0.0]).unsqueeze(1)
            data = PyG_Data(x=vector, edge_index=transform_edge_index(v, edge_index), vector=v.unsqueeze(0))
            data_list.append(data)
    else:
        for idx, v in enumerate(input_data):
            vector = torch.Tensor([0.0] + v.tolist() + [0.0]).unsqueeze(1)
            y = [target_data[idx] for target_data in target_data_lst]
            data = PyG_Data(x=vector, edge_index=transform_edge_index(v, edge_index), y=y, vector=v.unsqueeze(0))
            data_list.append(data)

    return data_list

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        self.agg = aggr.MLPAggregation(256, 32, 6, num_layers=1)
        self.v = nn.Linear(6, 256)
        self.classifier = nn.Sequential(nn.LayerNorm(32 + 256), nn.SiLU(),
                                        nn.Linear(32 + 256, 512), nn.LayerNorm(512),
                                        nn.SiLU(),
                                        nn.Linear(512, 512), nn.LayerNorm(512),
                                        nn.SiLU())

    def forward(self, x, edge_index, batch_index, vector):
        h_ = self.conv1(x, edge_index)
        h_ = h_.relu()
        h_ = self.conv2(h_, edge_index)
        h_ = h_.relu()
        h_ = self.conv3(h_, edge_index)
        h_ = h_.relu()
        h_ = self.agg(h_, batch_index)
        h_ = torch.cat([h_, self.v(vector)], dim=1)
        out = self.classifier(h_)
        return out


def encoder_generator_func_():
    return GCN()


def one_run(pretrain_epochs, total_evals, accel, threads, run_name, evals_filename, metrics_filename, bench_optimum, pretrain=True, plot_logs=True):
    # Ensemble instance and pretraining
    with open(metrics_filename, 'rb') as f:
        encodings, metrics = pickle.load(f)
        pretrain_metrics_pyg_list = get_pyg_data_list(encodings, metrics)

    e = EnsemblePyG(pretrain_metrics_pyg_list=pretrain_metrics_pyg_list, n_pretrain_metrics=3,
                    network_generator_func=encoder_generator_func_, embedding_dim=embedding_dim,
                    n_networks=6, accelerator=accel, devices=threads, train_lr=5e-3,
                    pretrain_epochs=pretrain_epochs, pretrain_lr=5e-3, pretrain_bs=8)

    print(f'Pretraining for {pretrain_epochs} epochs')
    if pretrain:
        e.pretrain()

    # Search
    with open(evals_filename, 'rb') as f:
        encodings, accuracies = pickle.load(f)
        all_pyg_data_list = get_pyg_data_list(encodings, [accuracies])

    s = BenchSearchPyG(experiment_name=run_name,
                       ensemble=e,
                       all_pyg_data_list=all_pyg_data_list,
                       is_optimum=lambda x: bool(x >= bench_optimum),
                       evals_per_iter=32,
                       epochs_per_iter=100)

    print(f'Search step. Total evals = {total_evals}.')
    logs = s.run(int(total_evals / 32))

    if plot_logs:
        s.plot_logs()

    print(f"Evals at which optimum was found:\t{logs[-1]['found_optimum_index']}\n")

    return logs


def multiple_runs(pretrain_epochs, total_evals, accel, threads, xp_name, evals_filename, metrics_filename, bench_optimum, pretrain=True, n_runs=10):
    logs_list = []
    for i in range(n_runs):
        logs_list.append(one_run(pretrain_epochs=pretrain_epochs,
                                 total_evals=total_evals, accel=accel, threads=threads,
                                 run_name=f'{xp_name}_{i}', evals_filename=evals_filename,
                                 metrics_filename=metrics_filename,
                                 bench_optimum=bench_optimum,
                                 pretrain=pretrain, plot_logs=False))
    return logs_list


def get_average_final_corr(lst):
    return np.array([l[-1]['correlations'][0].correlation for l in lst]).mean()


def get_average_index_optimum_reached(lst):
    indices = [l[-1]['found_optimum_index'] for l in lst]
    clean_list = [i for i in indices if i is not None]
    n_fails = indices.count(None)
    if len(clean_list) == 0:
        return None
    return np.mean(clean_list), n_fails


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', type=str, default='nb201_pretrained_gcn')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--pretraining', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--total-evals', type=int, default=1024)
    parser.add_argument('--pretrain-epochs', type=int, default=1000)
    parser.add_argument('--datasets', type=str, nargs='+', default=['cifar10'])
    args = parser.parse_args()

    pretrain_epochs, total_evals, accel, threads, xp_name = args.pretrain_epochs, args.total_evals, args.accelerator, args.threads, args.experiment_name
    n_runs, pretrain, datasets = args.runs, args.pretraining, args.datasets

    bench_optima = {'cifar10': 0.9437, 'cifar100': 0.7351, 'ImageNet16-120': 0.4731}

    logs = dict()
    for d in datasets:
        evals_filename = f'experiments/nasbench_201/pretraining_data/nats_tss_{d}_evals.pickle'
        metrics_filename = f'experiments/nasbench_201/pretraining_data/nats_tss_{d}_metrics.pickle'
        logs[d] = multiple_runs(pretrain_epochs=pretrain_epochs, total_evals=total_evals,
                                accel=accel, threads=threads,
                                xp_name=xp_name, evals_filename=evals_filename,
                                metrics_filename=metrics_filename,
                                bench_optimum=bench_optima[d],
                                pretrain=pretrain, n_runs=n_runs)

    snapshots = [100, 200, 400, 444]
    print('\nSummary')
    for d in datasets:
        print(f'{d}\t{n_runs} runs, {"With pretraining" if pretrain else "No pretraining"}')
        print('Acc = {}\tRho = {}\tIdx = {}\tF = {}'.format(*grab_data(logs[d])))
        print('Evals at optimum found: \tmin = {}\tmax = {}\tavg = {}\tstd = {}'.format(*get_index_stats(logs[d])))
        for s in snapshots:
            print('Snapshot at {}\t\tn_evals = {}\t\tacc = {}'.format(s, *get_values_at_x(logs[d], s)))
        print('\n')