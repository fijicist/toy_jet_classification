import os
import gc
import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv, global_mean_pool
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from jetnet.losses import EMDLoss

class GATHypergraphNet(torch.nn.Module):
    """
    A model that processes edge features using GATConv layers and hypergraph features
    using HypergraphConv layers, then combines the results for jet classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GATHypergraphNet, self).__init__()

        # GATConv layers for edge features
        self.gat_conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True, dropout=dropout_rate)
        self.gat_conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True, dropout=dropout_rate)

        # HypergraphConv layers for hyperedge features
        self.hyper_conv1 = HypergraphConv(in_channels, hidden_channels, use_attention=True, attention_mode='edge', heads=4)
        self.hyper_conv2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=True, attention_mode='edge', heads=4)

        # Fully connected layers for classification
        self.lin1 = nn.Linear(hidden_channels * 8, hidden_channels)  # Combine GAT and Hypergraph outputs
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        # Extract node features (x), edge index (edge_index), and hyperedge index (hyperedge_index)
        x, edge_index, edge_features, hyperedge_index, hyperedge_features = data.x, data.edge_index, data.edge_attr, data.hyperedge_index, data.hyperedge_attr

        # Process edge features with GATConv layers
        x_gat = self.gat_conv1(x, edge_index=edge_index, edge_attr=edge_features)
        x_gat = F.relu(x_gat)
        x_gat = self.dropout(x_gat)
        x_gat = self.gat_conv2(x_gat, edge_index=edge_index, edge_attr=edge_features)
        x_gat = F.relu(x_gat)
        x_gat = self.dropout(x_gat)

        # Process hyperedge features with HypergraphConv layers
        x_hyper = self.hyper_conv1(x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_features)
        x_hyper = F.relu(x_hyper)
        x_hyper = self.dropout(x_hyper)
        x_hyper = self.hyper_conv2(x_hyper, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_features)
        x_hyper = F.relu(x_hyper)
        x_hyper = self.dropout(x_hyper)

        # Combine GAT and Hypergraph outputs
        x_combined = torch.cat([x_gat, x_hyper], dim=1)

        # Global pooling to reduce each graph into a feature vector
        x_pooled = global_mean_pool(x_combined, data.batch)

        # Fully connected layers for classification
        x_out = F.relu(self.lin1(x_pooled))
        x_out = self.dropout(x_out)
        x_out = self.lin2(x_out)

        return F.softmax(x_out, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, n_input_features, hidden_dim, n_output_classes, dropout_rate=0.1):
        super(GAT, self).__init__()
        self.head = 8

        self.conv1 = GATConv(n_input_features, hidden_dim, heads=self.head, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * self.head, hidden_dim, heads=self.head, dropout=dropout_rate)

        # Dropout layer (by default, only active during training -- i.e. disabled with mode.eval() )
        self.dropout = nn.Dropout(p=dropout_rate)

        self.lin = nn.Linear(hidden_dim * self.head, n_output_classes)

        # testing a linear layer before gatconv layer
        self.lin1 = nn.Linear(n_input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.lin3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv3 = GATConv(128, hidden_dim, heads=self.head, dropout=dropout_rate)

    def forward(self, x, edge_index, batch):
        
        # testing a linear layer before gatconv layer
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.lin3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # GNN layers
        x = self.conv3(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        # testing another hidden layer
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        # testing another hidden layer
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        # # testing another hidden layer
        # x = self.conv2(x, edge_index)
        # x = F.gelu(x)
        # x = self.dropout(x)
        #
        # # testing another hidden layer
        # x = self.conv2(x, edge_index)
        # x = F.gelu(x)
        # x = self.dropout(x)
        #
        # # testing another hidden layer 
        # x = self.conv2(x, edge_index)
        # x = F.gelu(x)
        # x = self.dropout(x)
        #
        # # testing another hidden layer 
        # x = self.conv2(x, edge_index)
        # x = F.gelu(x)
        # x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return F.softmax(x, dim=1)

# class GAT(torch.nn.Module):
#     def __init__(self, n_input_features, hidden_dim, n_output_classes, dropout_rate=0.25):
#         super(GAT, self).__init__()
#         self.head = 8
#
#         self.conv1 = GATConv(n_input_features, hidden_dim, heads=self.head, dropout=dropout_rate)
#         self.conv2 = GATConv(hidden_dim * self.head, hidden_dim, heads=self.head, dropout=dropout_rate)
#
#         # Dropout layer (by default, only active during training -- i.e. disabled with mode.eval() )
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#         self.lin = nn.Linear(hidden_dim * self.head, n_output_classes)
#
#         # testing a linear layer before gatconv layer
#         self.lin1 = nn.Linear(n_input_features, 128)
#         self.lin2 = nn.Linear(128, 512)
#         self.lin3 = nn.Linear(512, 128)
#         self.conv3 = GATConv(128, hidden_dim, heads=self.head, dropout=dropout_rate)
#
#     def forward(self, x, edge_index, batch):
#         
#         # testing a linear layer before gatconv layer
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.lin2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.lin3(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # GNN layers
#         # x = self.conv1(x, edge_index)
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # testing another hidden layer 
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # testing another hidden layer 
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # testing another hidden layer
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # testing another hidden layer
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # testing another hidden layer 
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # testing another hidden layer 
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = global_mean_pool(x, batch)
#
#         x = self.lin(x)
#
#         return F.softmax(x, dim=1)

class GCNModel(nn.Module):
    def __init__(self, n_input_features, hidden_dim, n_output_classes, dropout_rate=0.5):
        super(GCNModel, self).__init__()

        # GNN layers
        self.conv1 = GCNConv(n_input_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Dropout layer (by default, only active during training -- i.e. disabled with mode.eval() )
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully connected layer for graph classification
        self.fc = nn.Linear(hidden_dim, n_output_classes)

    def forward(self, x, edge_index, batch):

        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling (i.e. avg node features across each graph) to get a graph-level representation for graph classification
        # This requires the batch tensor, which keeps track of which nodes belong to which graphs in the batch.
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        # Note: For now, we don't apply dropout here, since the dimension is small
        x = self.fc(x)

        return F.softmax(x, dim=1)

class MLAnalysis:
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 model="GCN",
                 batch_size=1024,
                 learning_rate=0.01,
                 epochs=100,
                 model_output_path='./trained_models',
                 metrics_plot_path='./metrics_plot'
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_output_path = model_output_path
        self.metrics_plot_path = metrics_plot_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model == "GCN":
            self.model = GCNModel(input_dim, hidden_dim, output_dim).to(self.device)
        elif model == "GAT":
            self.model = GAT(input_dim, hidden_dim, output_dim).to(self.device)
        elif model == "GATHyper":
            self.model = GATHypergraphNet(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = np.zeros(epochs)
        self.test_losses = np.zeros(epochs)
        self.train_accuracies = np.zeros(epochs)
        self.test_accuracies = np.zeros(epochs)
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        print("Loading data...")
        print(self.device)
        dataset_parts = []
        folder_path = './graph_objects/particle_graphs/'
        for file_name in tqdm(os.listdir(folder_path), desc="Loading files"):
            if file_name.endswith('.pt'):
                part = torch.load(os.path.join(folder_path, file_name))
                dataset_parts += part
                # Free memory after loading each .pt file 
                del part
                gc.collect()
        
        dataset_size = len(dataset_parts)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset_parts, [train_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        del dataset_parts
        del train_dataset
        del test_dataset
        gc.collect()

        for step, data in enumerate(self.train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
        print("Data loading completed.")

    def accuracy(self, loader):
        correct = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model.forward(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        accuracy = correct / len(loader.dataset)
        return accuracy

    def train(self):
        print("Starting training...")
        
        for epoch in tqdm(range(self.epochs), desc="Training epochs"):
            for data in self.train_loader:
                data = data.to(self.device)
                if self.model.__class__.__name__ == "GATHypergraphNet":
                    out = self.model.forward(data)
                else:
                    out = self.model.forward(data.x, data.edge_index, data.batch)
                loss = self.criterion(out, data.y.long())
                self.train_losses[epoch] += loss.item()
                self.train_losses[epoch] /= len(self.train_loader)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            for test_data in self.test_loader:
                test_data = test_data.to(self.device)
                out = self.model.forward(test_data.x, test_data.edge_index, test_data.batch)
                test_loss = self.criterion(out, test_data.y.long())
                self.test_losses[epoch] += test_loss.item()
                self.test_losses[epoch] /= len(self.test_loader)

            train_acc = self.accuracy(self.train_loader)
            test_acc = self.accuracy(self.test_loader)
            self.train_accuracies[epoch] = train_acc
            self.test_accuracies[epoch] = test_acc

            print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print("Training completed.")

        # Save model
        if not os.path.exists(self.model_output_path):
            os.makedirs(self.model_output_path)
            print(f"Directory {self.model_output_path} created.")
        else:
            print(f"Directory {self.model_output_path} already exists.")

        torch.save(self.model.state_dict(), os.path.join(self.model_output_path,\
        f'{self.input_dim}_{self.hidden_dim}_{self.model.__class__.__name__}_{self.batch_size}_{self.learning_rate}_{self.epochs}.pt'))


    def evaluate(self):
        print("Starting evaluation...")

        # Evaluate model on test set
        pred_graphs_list = []
        label_graphs_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                pred_graph = self.model.forward(batch.x, batch.edge_index, batch.batch)
                pred_graphs_list.append(pred_graph.cpu().data.numpy())
                label_graphs_list.append(batch.y.cpu().data.numpy())
            pred_graphs = np.concatenate(pred_graphs_list, axis=0)
            label_graphs = np.concatenate(label_graphs_list, axis=0)
            auc = metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
            roc_curve = metrics.roc_curve(label_graphs, pred_graphs[:,1])

            print(f'Evaluation completed. AUC: {auc:.4f}')
            return auc, roc_curve

    def display_metrics(self):
        epochs = range(1, self.epochs + 1)
        
        plt.figure(figsize=(18, 4))

        # First subplot for loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Second subplot for accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        #Third subplot for ROC curve
        plt.subplot(1, 3, 3)

        # Compute fpr, tpr, thresholds and roc auc
        auc, roc_curve = self.evaluate()
        fpr, tpr, thresholds = roc_curve

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.tight_layout()

        if not os.path.exists(self.metrics_plot_path):
            os.makedirs(self.metrics_plot_path)
            print(f"Directory {self.metrics_plot_path} created.")
        else:
            print(f"Directory {self.metrics_plot_path} already exists.")

        plt.savefig("./metrics_plot/metrics_plot"+"_"+str(self.input_dim)+"_"+\
            str(self.hidden_dim)+"_"+str(self.model.__class__.__name__)+"_"+str(self.batch_size)+"_"+str(self.learning_rate)+".png")

analysis = MLAnalysis(7, 32, 2, model="GATHyper", batch_size=256, learning_rate=0.0001, epochs=100)

analysis.load_data()

analysis.train()
analysis.display_metrics()
