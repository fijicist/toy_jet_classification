import os
import gc
import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class GAT(torch.nn.Module):
    def __init__(self, n_input_features, hidden_dim, n_output_classes, dropout_rate=0.5):
        super(GAT, self).__init__()
        self.head = 8

        self.conv1 = GATConv(n_input_features, hidden_dim, heads=self.head, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * self.head, hidden_dim, heads=self.head, dropout=dropout_rate)

        # Dropout layer (by default, only active during training -- i.e. disabled with mode.eval() )
        self.dropout = nn.Dropout(p=dropout_rate)

        self.lin = nn.Linear(hidden_dim * self.head, n_output_classes)

    def forward(self, x, edge_index, batch):
        
        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
    
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return F.softmax(x, dim=1)

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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

        print(self.test_loader, self.train_loader)
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

analysis = MLAnalysis(4, 8, 2, model="GAT", batch_size=128, learning_rate=0.0001, epochs=20)

analysis.load_data()

analysis.train()
analysis.display_metrics()
