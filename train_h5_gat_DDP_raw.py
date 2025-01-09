import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import time
import h5py
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

def load_data_from_hdf5(hdf5_filename):
    data_list = []
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        for molecule_key in hdf5_file.keys():
            molecule_group = hdf5_file[molecule_key]
            
            coordinates = torch.tensor(molecule_group['coordinates'][:], dtype=torch.float)
            bonds = molecule_group['bonds'][:]
            # energy_normalizeの代わりにenergy_kcalを使用
            energy = torch.tensor(molecule_group['energy_kcal'][()], dtype=torch.float).view(-1, 1)

            edge_index = []
            edge_attr = []
            for bond in bonds:
                atom1, atom2, bond_strength = int(bond[0]) - 1, int(bond[1]) - 1, bond[2]
                edge_index.append([atom1, atom2])
                edge_attr.append([bond_strength])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            data = Data(x=coordinates, edge_index=edge_index, edge_attr=edge_attr, y=energy)
            data_list.append(data)
    return data_list

class GCN(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, conv_type, dropout_rate, activation_function, dense_units):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_type(3, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(conv_type(hidden_channels, hidden_channels))
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(hidden_channels, dense_units)
        self.fc2 = torch.nn.Linear(dense_units, 1)
        self.activation = activation_function

    def forward(self, data):
        device = data.x.device
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
            
        x = global_mean_pool(x, batch)
        x = x.to(self.fc1.weight.device)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def weight_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            m.lin.bias.data.fill_(0.01)
    elif isinstance(m, GATConv):
        if hasattr(m, 'lin_src') and m.lin_src is not None:
            torch.nn.init.xavier_uniform_(m.lin_src.weight)
            if m.lin_src.bias is not None:
                m.lin_src.bias.data.fill_(0.01)
        if hasattr(m, 'lin_dst') and m.lin_dst is not None:
            torch.nn.init.xavier_uniform_(m.lin_dst.weight)
            if m.lin_dst.bias is not None:
                m.lin_dst.bias.data.fill_(0.01)
    elif isinstance(m, SAGEConv):
        if hasattr(m, 'lin_l') and m.lin_l is not None:
            torch.nn.init.xavier_uniform_(m.lin_l.weight)
            if m.lin_l.bias is not None:
                m.lin_l.bias.data.fill_(0.01)
        if hasattr(m, 'lin_r') and m.lin_r is not None:
            torch.nn.init.xavier_uniform_(m.lin_r.weight)
            if m.lin_r.bias is not None:
                m.lin_r.bias.data.fill_(0.01)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, config):
    setup(rank, world_size)
    wandb.init(project="gat_coordinate_normalized_conformer", config=config)
    config = wandb.config

    # データの準備
    hdf5_filename = '/home/gemma/ml/TSenergy/experiment/dataset_40atoms/h5data/coordinate_normalized_conformer.h5'
    external_hdf5_filename = '/home/gemma/ml/TSenergy/experiment/dataset_40atoms/h5data/test_energy_normalized_conformer.h5'
    data_list = load_data_from_hdf5(hdf5_filename)
    external_test_data = load_data_from_hdf5(external_hdf5_filename)
    train_data, val_data = train_test_split(data_list, test_size=0.4, random_state=42)

    # DataLoaderの作成 (DistributedSamplerを使用)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, sampler=val_sampler)
    external_test_loader = DataLoader(external_test_data, batch_size=config.batch_size, shuffle=False)

    # モデルの初期化
    device = torch.device(f'cuda:{rank}')
    model = GCN(
        config.n_layers,
        config.hidden_channels,
        GCNConv,  # 使用する畳み込みタイプ
        config.dropout_rate,
        F.relu,  # 活性化関数
        config.dense_units
    ).to(device)
    model = DDP(model, device_ids=[rank])

    # 重みの初期化
    model.apply(weight_init)

    # 最適化関数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.L1Loss()

    # 学習ループ
    train_losses = []
    val_losses = []
    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)  # 各エポックでデータシャッフル
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_graphs

        train_loss = loss_all / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 検証フェーズ
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y)
                val_loss += loss.item() * data.num_graphs

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # 外部テストデータ
        extest_loss = 0
        with torch.no_grad():
            for data in external_test_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y)
                extest_loss += loss.item() * data.num_graphs

        extest_loss = extest_loss / len(external_test_loader.dataset)


        end_time = time.time()
        epoch_time = end_time - start_time
        # 安定性スコアとカスタムメトリックの計算
        num_epochs_to_consider = 20
        stability_score = np.std(val_losses[-num_epochs_to_consider:]) if len(val_losses) >= num_epochs_to_consider else np.std(val_losses)
        custom_metric = val_loss + 1.5 * stability_score

        # rank 0 のみ出力とログ
        if rank == 0:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "extest_loss": extest_loss,
                "stability_score": stability_score,
                "custom_metric": custom_metric,
                "epoch": epoch
            })
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, External Test Loss: {extest_loss:.4f}, Epoch {epoch + 1} took {epoch_time:.2f} seconds")
    
    cleanup()

def train(rank, world_size, config_dict):
    if rank == 0:
        # 主プロセスのみ wandb を初期化
        wandb.init(
            project="gat_coordinate_normalized_conformer",
            config=config_dict,
            reinit=True
        )
        config = wandb.config
    else:
        config = config_dict  # 他のプロセスは config_dict をそのまま使用

    train_ddp(rank, world_size, config)

def main():
    # メインプロセスで wandb のスイープ設定を取得し、辞書に変換
    wandb.init()
    config_dict = dict(wandb.config)  # wandb.config を辞書に変換して各プロセスに渡す
    
    # 利用可能な GPU 数を取得し、分散学習の設定
    world_size = torch.cuda.device_count()

    # train 関数をマルチプロセスで実行
    mp.spawn(train, args=(world_size, config_dict), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

