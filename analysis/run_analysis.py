from utils import get_eec_ls_values
from graph_constructor import _construct_particle_graphs_pyg, _construct_particle_graphs_pyg
from ml_analysis import GCNModel, MLAnalysis


### For particle graphs construction ###

graph_output_dir = "./graph_objects/particle_graphs/"
graph_structure = ['fully_connected']
num_jets = 100000
eec_prop = [2, 50, (1e-3, 1)] # [N, bins, (R_Lmin, R_Lmax)]j

additional_node_attrs=None,
additional_edge_attrs=None, # None or eec_with_charges or eec_without_charges
additional_graph_attrs=None,
additional_hypergraph_attrs=None

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

#########################################

### For running jet classification task with GCNModel ###

input_dim = 3 # no. of node features or input features
hidden_dim = 16 # no. of hidden features
output_dim = 2 # no. of output features

batch_size = 1024
learning_rate = 0.0003
epochs = 100

model_output_path = './trained_models'
metrics_plot_path = './metrics_plot'

analysis = MLAnalysis(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
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
