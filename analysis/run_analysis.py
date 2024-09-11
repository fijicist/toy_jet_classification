import os
from utils import get_eec_ls_values
from graph_constructor import _construct_particle_graphs_pyg, _construct_particle_graphs_pyg
from ml_analysis import GCNModel, MLAnalysis


### For particle graphs construction ###

graph_output_dir = "./graph_objects/particle_graphs/"
graph_structure = ['fully_connected']
num_jets = 100000
eec_prop = [2, 50, (1e-3, 2)] # [N, bins, (R_Lmin, R_Lmax)]j

additional_node_attrs=None,
additional_edge_attrs='eec_with_charges', # None or eec_with_charges or eec_without_charges
additional_graph_attrs=None,
additional_hypergraph_attrs=None

force_create = False

def particle_graphs_exist(directory):
    # Check if the directory exists and contains files
    return os.path.exists(directory) and len(os.listdir(directory)) > 0

if not particle_graphs_exist(graph_output_dir) or force_create:
    _construct_particle_graphs_pyg(
        output_dir=graph_output_dir,
        graph_structures=graph_structure,
        N=num_jets,
        eec_prop=eec_prop,
        additional_node_attrs=additional_node_attrs,
        additional_edge_attrs=additional_edge_attrs,
        additional_graph_attrs=additional_graph_attrs,
        additional_hypergraph_attrs=additional_hypergraph_attrs
    )
else:
    print("Particle graphs already exist. Skipping creation.")

#########################################

### For running jet classification task ###

input_dim = 4 # no. of node features or input features
hidden_dim = 8 # no. of hidden features
output_dim = 2 # no. of output features

batch_size = 256
learning_rate = 0.0001
epochs = 50

model = "GAT" # "GCN" or "GAT"

model_output_path = './trained_models'
metrics_plot_path = './metrics_plot'

analysis = MLAnalysis(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    model=model,
    batch_size=batch_size,
    learning_rate=learning_rate,
    epochs=epochs,
    model_output_path=model_output_path,
    metrics_plot_path=metrics_plot_path
)

analysis.load_data()
analysis.train()
analysis.display_metrics()

#########################################################
