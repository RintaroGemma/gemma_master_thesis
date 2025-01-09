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

def parse_bond_info(bond_info):
    parsed_bonds = []

    for bond_line in bond_info:
        if bond_line.strip() == "":
            continue
        
        bond_elements = bond_line.split()
        atom1 = int(bond_elements[0])
        
        for i in range(1, len(bond_elements), 2):
            if i + 1 < len(bond_elements):
                atom2 = int(bond_elements[i])
                bond_strength = float(bond_elements[i + 1])
                parsed_bonds.append((atom1, atom2, bond_strength))
    
    return parsed_bonds

def create_pyg_data(atomic_coordinates, parsed_bonds, energy_normalized):
    coordinates = []
    for line in atomic_coordinates:
        elements = line.split()
        coordinates.append([float(elements[1]), float(elements[2]), float(elements[3])])

    x = torch.tensor(coordinates, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for bond in parsed_bonds:
        atom1, atom2, bond_strength = bond
        edge_index.append([atom1 - 1, atom2 - 1])  # 0ベースに変換
        edge_attr.append([bond_strength])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([energy_normalized], dtype=torch.float).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

def process_directories(input_dirs):
    data_list = []
    for input_dir in input_dirs:
        file_paths = glob.glob(os.path.join(input_dir, '*.txt'))
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            print(f"Processing {filename} in {input_dir}...")
            
            energy_info, atomic_coordinates, bond_info = read_molecular_file(file_path)
            parsed_bonds = parse_bond_info(bond_info)
            
            energy_normalized = None
            for line in energy_info:
                if "Energy(normalize)" in line:
                    energy_normalized = float(line.split(":")[1].strip())
                    break
            
            if energy_normalized is not None:
                pyg_data = create_pyg_data(atomic_coordinates, parsed_bonds, energy_normalized)
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
wandb.init(project="GAT_energy_normalized_Gshifted")

# ハイパーパラメータのログ
config = wandb.config
config.hidden_channels = wandb.config.get("hidden_channels", 128)
config.optimizer = wandb.config.get("optimizer", "adam")
config.learning_rate = wandb.config.get("learning_rate", 0.0001)
config.n_layers = wandb.config.get("n_layers", 2)
config.knn_k = wandb.config.get("knn_k", 2)
config.n_units = wandb.config.get("n_units", 256)
config.feat_type = wandb.config.get("feat_type", "gat")  # 修正された値
config.epochs = wandb.config.get("epochs", 500)
config.batch_size = wandb.config.get("batch_size", 8)
config.dropout_rate = wandb.config.get("dropout_rate", 0.0)
config.activation_function = wandb.config.get("activation_function", "relu")
config.dense_units = wandb.config.get("dense_units", 1)

# Convレイヤータイプのマッピング
conv_layer_mapping = {
    "gcn": GCNConv,
    "gat": GATConv,  # 修正された値
    "sage": SAGEConv
}

# 活性化関数のマッピング
activation_function_mapping = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu
}

# 入力ディレクトリのリストを指定してください
input_dirs = [
    #'/homg/gemma/ml/TSenergy/40atoms_files_rotate',
    '/home/gemma/ml/TSenergy/experiment/dataset_40atoms/data_energy_normalized_Gshifted',
    #'/home/gemma/ml/TSenergy/40atoms_files',
    #'/home/gemma/ml/TSenergy/53atoms_files',
    #'/home/gemma/ml/TSenergy/66atoms_files'
]

# 外部テストデータのロード
external_data_path = '/home/gemma/ml/TSenergy/experiment/dataset_40atoms/testdata_energy_normalized_shiftG'  # 外部テストデータのパスに置き換えてください
external_data_list = process_directories([external_data_path])
external_test_loader = DataLoader(external_data_list, batch_size=config.batch_size, shuffle=False)


# 複数のディレクトリ内のすべてのテキストファイルを処理
data_list = process_directories(input_dirs)

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
    start_time = time.time()  # エポック開始時の時間を記録

    model.train()
    loss_all = 0
    train_preds = []
    val_preds = []
    test_preds = []
    extest_preds = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        train_preds.append(output.cpu().detach().numpy())

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
            val_preds.append(output.cpu().detach().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # テストデータに対する損失を計算
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            test_loss += loss.item() * data.num_graphs
            test_preds.append(output.cpu().detach().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(test_loss)
    
    # 外部テストデータに対する損失を計算
    extest_loss = 0
    with torch.no_grad():
        for data in external_test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            extest_loss += loss.item() * data.num_graphs
            extest_preds.append(output.cpu().detach().numpy())

    extest_loss = extest_loss / len(external_test_loader.dataset)
    extest_losses.append(extest_loss)

    # エポック終了時の時間を計測
    end_time = time.time()
    epoch_time = end_time - start_time

    # 損失、学習率、予測結果、重みの分布をwandbにログ
    if train_preds:
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "extest_loss": extest_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time,  # 追加
            "weights": wandb.Histogram(model.fc1.weight.data.cpu().numpy()),
            "gradients": wandb.Histogram(model.fc1.weight.grad.data.cpu().numpy()),
            "train_predictions": wandb.Histogram(np.concatenate(train_preds)),
            "val_predictions": wandb.Histogram(np.concatenate(val_preds)),
            "test_predictions": wandb.Histogram(np.concatenate(test_preds))
        })

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, External Test Loss: {extest_loss:.4f}')

    # test_lossが0.0002を下回った場合、Sweepを終了する
    if test_loss < 0.0002:
        print("test_lossが0.0002を下回りました。Sweepを終了します。")
        wandb.log({"test_loss_below_threshold": test_loss})
        wandb.finish()
        wandb.agent.stop()  # Sweepを終了させる
        break

# テストデータに対する損失を計算
model.eval()
test_loss = 0
test_preds = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.y)
        test_loss += loss.item() * data.num_graphs
        test_preds.append(output.cpu().detach().numpy())

test_loss = test_loss / len(test_loader.dataset)
wandb.log({"final_test_loss": test_loss, "test_predictions": wandb.Histogram(np.concatenate(test_preds))})
print(f'Final Test Loss: {test_loss:.4f}')

# wandbにモデルを保存
torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

