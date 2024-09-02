"""
The graph_constructor module constructs the input graphs to the ML analysis:
    - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
"""

import os
import tqdm
import numpy as np
import energyflow
import torch
import torch_geometric

from import_dataset import import_CMS2011AJets_dataset
from utils import get_eec_ls_values

def _construct_particle_graphs_pyg(
        output_dir,
        graph_structures,
        N=500000,
        eec_prop=[2, 50, (1e-3, 1)], # [N, bins, (R_Lmin, R_Lmax)]
        additional_node_attrs=None,
        additional_edge_attrs=None, # None or eec_with_charges or eec_without_charges
        additional_graph_attrs=None,
        additional_hypergraph_attrs=None
):
    '''
    Construct a list of PyG graphs for the particle-based GNNs, loading from the energyflow dataset

    Graph structure:
        - Nodes: particle four-vectors
        - Edges: no edge features
        - Connectivity: fully connected 
    '''
    print(f'Constructing PyG particle graphs from energyflow dataset...')

    # Load dataset
    if not additional_node_attrs and not additional_edge_attrs and not additional_graph_attrs and not additional_hypergraph_attrs:
        X, y = energyflow.qg_jets.load(N)
        X = X[:,:,:3]
        print(f'  (n_jets, n_particles, features): {X.shape}')

    elif additional_edge_attrs == 'eec_with_charges':
        X, y = energyflow.qg_jets.load(N, pad=False)
        for i in range(len(X)):
            X[i][:, 3] = energyflow.pids2chrgs(X[i][:, 3])

    elif additional_edge_attrs == 'eec_without_charges':
        X, y = energyflow.qg_jets.load(N, pad=False)
        for i in range(len(X)):
            X[i] = X[i][:, :3]


    # Preprocess by centering jets and normalizing pts
    for x in tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X)):
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()

    # Calculate EnergyEnergyCorrelation (EEC) features
    if additional_edge_attrs == 'eec_with_charges':
        print(f'  Calculating EEC features with charges...')
        additional_edge_attrs = get_eec_ls_values(X, N=eec_prop[0], bins=eec_prop[1], axis_range=eec_prop[2])

    if additional_edge_attrs == 'eec_without_charges':
        print(f'  Calculating EEC features without charges...')
        additional_edge_attrs = get_eec_ls_values(X, N=eec_prop[0], bins=eec_prop[1], axis_range=eec_prop[2])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    for graph_structure in graph_structures:
        graph_key = f'particle__{graph_structure}'

        graph_list = []
        args = [(x, y[i]) for i, x in enumerate(X)]
        for i, arg in enumerate(tqdm.tqdm(args, desc=f'  Constructing PyG graphs: {graph_key}', total=len(args))):
            graph_list.append(_construct_particle_graph_pyg(arg, additional_node_attrs, additional_edge_attrs, additional_graph_attrs, additional_hypergraph_attrs))

            # Save to file every 100,000 iterations
            if (i + 1) % 100000 == 0:
                partial_graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}_part_{i // 100000 + 1}.pt")
                torch.save(graph_list, partial_graph_filename)
                print(f'  Saved PyG graphs to {partial_graph_filename}.')
                graph_list = []

        # Save any remaining graphs
        if graph_list:
            final_graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}_final.pt")
            torch.save(graph_list, final_graph_filename)
            print(f'  Saved PyG graphs to {final_graph_filename}.')

def _construct_particle_graph_pyg(
        args,
        additional_node_attrs=None,
        additional_edge_attrs=None,
        additional_graph_attrs=None,
        additional_hypergraph_attrs=None
):
    '''
    Construct a single PyG graph for the particle-based GNNs from the energyflow dataset
    '''
    x, label = args

    # Node features -- remove the zero pads
    x = x[~np.all(x == 0, axis=1)]
    node_features = torch.tensor(x, dtype=torch.float)

    # Edge connectivity -- fully connected
    adj_matrix = np.ones((x.shape[0], x.shape[0])) - np.identity((x.shape[0]))
    row, col = np.where(adj_matrix)
    coo = np.array(list(zip(row, col)))
    edge_indices = torch.tensor(coo)
    edge_indices_long = edge_indices.t().to(torch.long).view(2, -1)
    print(edge_indices_long, edge_indices_long.shape)

    # Construct graph as PyG data object
    graph_label = torch.tensor(label, dtype=torch.bool)

    # Add additional attributes if provided
    if additional_node_attrs:
        graph.node_attrs = torch.tensor(additional_node_attrs, dtype=torch.float)
    if additional_edge_attrs:
        # Calculate edge features
        edge_features = []
        for i, j in zip(row, col):
            edge_value = np.sqrt(x[i][1]**2 + x[i][2]**2)
            # Determine the bin for the edge value
            bin_index = np.digitize(edge_value, bins=additional_edge_attrs['bin_edges']) - 1
            # Get the histogram value for the bin
            edge_feature = additional_edge_attrs['histogram'][bin_index]
            edge_features.append(edge_feature)

        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)
        graph.edge_attrs = torch.tensor(additional_edge_attrs, dtype=torch.float)
    if additional_graph_attrs:
        graph.graph_attrs = torch.tensor(additional_graph_attrs, dtype=torch.float)
    if additional_hypergraph_attrs:
        graph.hypergraph_attrs = torch.tensor(additional_hypergraph_attrs, dtype=torch.float)
    else:
        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=None, y=graph_label)

    return graph

#_construct_particle_graphs_pyg("./graph_objects/particle_graphs/.", ['fully_connected'], 100000)

_construct_particle_graphs_pyg("./graph_objects/particle_graphs/.", ['fully_connected'], 100000, additional_edge_attrs='eec_without_charges')
