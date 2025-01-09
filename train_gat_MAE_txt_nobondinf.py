import os
import glob
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import wandb
import time

def read_molecular_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    energy_info = []
    atomic_coordinates = []
    bond_info = []

    section = 0

    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "":
            section += 1
            continue

        if section == 0:
            energy_info.append(stripped_line)
        elif section == 1:
            atomic_coordinates.append(stripped_line)
        elif section == 2:
            bond_info.append(stripped_line)

    return energy_info, atomic_coordinates, bond_info

def create_pyg_data_without_bonds(atomic_coordinates, energy_normalized):
    coordinates = []
    for line in atomic_coordinates:
        elements = line.split()
        coordinates.append([float(elements[1]), float(elements[2]), float(elements[3])])

    x = torch.tensor(coordinates, dtype=torch.float)

    # エッジ情報を削除（Noneに設定）
    edge_index = None
    edge_attr = None

    y = torch.tensor([energy_normalized], dtype=torch.float).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

def process_directories_without_bonds(input_dirs):
    data_list = []
    for input_dir in input_dirs:
        file_paths = glob.glob(os.path.join(input_dir, '*.txt'))
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            print(f"Processing {filename} in {input_dir}...")
            
            energy_info, atomic_coordinates, bond_info = read_molecular_file(file_path)
            
            energy_normalized = None
            for line in energy_info:
                if "Energy(normalize)" in line:
                    energy_normalized = float(line.split(":")[1].strip())
                    break
            
            if energy_normalized is not None:
                pyg_data = create_pyg_data_without_bonds(atomic_coordinates, energy_normalized)
                data_list.append(pyg_data)

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
        x, edge_index = data.x, data.edge_index

        # データをデバイスに移動
        device = x.device
        if edge_index is None:
            edge_index = torch.tensor([[], []], dtype=torch.long).to(device)  # ダミーのエッジ情報をデバイスに移動

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, data.batch)
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
        torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            m.lin.bias.data.fill_(0.01)

# wandbを初期化
wandb.init(project="GAT_energy_normalized_nobondinfo_mod")

# ハイパーパラメータのログ
config = wandb.config
config.hidden_channels = wandb.config.get("hidden_channels", 128)
config.optimizer = wandb.config.get("optimizer", "adam")
config.learning_rate = wandb.config.get("learning_rate", 0.0001)
config.n_layers = wandb.config.get("n_layers", 2)
config.knn_k = wandb.config.get("knn_k", 2)
config.n_units = wandb.config.get("n_units", 256)
config.feat_type = wandb.config.get("feat_type", "gat")
config.epochs = wandb.config.get("epochs", 500)
config.batch_size = wandb.config.get("batch_size", 8)
config.dropout_rate = wandb.config.get("dropout_rate", 0.0)
config.activation_function = wandb.config.get("activation_function", "relu")
config.dense_units = wandb.config.get("dense_units", 1)

# Convレイヤータイプのマッピング
conv_layer_mapping = {
    "gcn": GCNConv,
    "gat": GATConv,
    "sage": SAGEConv
}

# 活性化関数のマッピング
activation_function_mapping = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu
}

# 入力ディレクトリのリストを指定してください
input_dirs = [
    '/home/gemma/ml/TSenergy/experiment/dataset_40atoms/energy_normalized'
]

# 外部テストデータのロード
external_data_path = '/home/gemma/ml/TSenergy/experiment/dataset_40atoms/testdata_energy_normalized'
external_data_list = process_directories_without_bonds([external_data_path])
external_test_loader = DataLoader(external_data_list, batch_size=config.batch_size, shuffle=False)

# データの処理
data_list = process_directories_without_bonds(input_dirs)

# データを訓練データセット、検証データセット、テストデータセットに分割
train_data, temp_data = train_test_split(data_list, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# DataLoaderを作成
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

# モデル、最適化関数、損失関数を定義
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(config.n_layers, config.hidden_channels, conv_layer_mapping[config.feat_type], config.dropout_rate, activation_function_mapping[config.activation_function], config.dense_units).to(device)

# 重みの初期化
model.apply(weight_init)

if config.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
elif config.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
elif config.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

criterion = torch.nn.L1Loss()

# 学習ループ
epochs = config.epochs
train_losses = []
val_losses = []
test_losses = []
extest_losses = []

for epoch in range(epochs):
    start_time = time.time()

    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    train_loss = loss_all / len(train_loader.dataset)
    train_losses.append(train_loss)

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
    
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            test_loss += loss.item() * data.num_graphs

    test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(test_loss)
    
    extest_loss = 0
    with torch.no_grad():
        for data in external_test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            extest_loss += loss.item() * data.num_graphs

    extest_loss = extest_loss / len(external_test_loader.dataset)
    extest_losses.append(extest_loss)

    end_time = time.time()
    epoch_time = end_time - start_time

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "extest_loss": extest_loss,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epoch_time": epoch_time
    })

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, External Test Loss: {extest_loss:.4f}')

    if test_loss < 0.0002:
        print("Test loss below 0.0002, stopping training.")
        wandb.log({"test_loss_below_threshold": test_loss})
        wandb.finish()
        break

model.eval()
test_loss = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.y)
        test_loss += loss.item() * data.num_graphs

test_loss = test_loss / len(test_loader.dataset)
wandb.log({"final_test_loss": test_loss})
print(f'Final Test Loss: {test_loss:.4f}')

torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

