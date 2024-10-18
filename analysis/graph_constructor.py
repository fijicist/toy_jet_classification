"""
The graph_constructor module constructs the input graphs to the ML analysis:
    - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
"""

import os
import gc
import tqdm
import numpy as np
import energyflow
import torch
import torch_geometric

import fastjet
import vector
import awkward as ak

from import_dataset import import_CMS2011AJets_dataset
from utils import get_eec_ls_values, plot_jet_kinematics, reclusterJets, ms2pids

def _construct_particle_graphs_pyg(
        output_dir,
        graph_structures,
        N=500000,
        eec_prop=[[2, 3, 4], 50, (1e-3, 2)], # [N, bins, (R_Lmin, R_Lmax)]
        additional_node_attrs=None,
        additional_edge_attrs=None, # None or eec_with_pids or eec_with_charges or eec_without_charges
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
    X, y = energyflow.qg_jets.load(N, pad=False)
    # energyflow.utils.remap_pids(X)

    # Reclustering the jets using fastjet to check the clustering
    print(f'  Reclustering jets using fastjet...')
    
    inclusive_jets = [[0, 0, 0, 0] for _ in range(len(X))]

    for i in range(len(X)):
        X[i] = energyflow.p4s_from_ptyphipids(X[i])
        X[i] = X[i].astype(np.float64)
    
        # Input to fastjet as an awkward array
        particleAwk = ak.zip({"px": X[i][:, 1], "py": X[i][:, 2], "pz": X[i][:, 3], "E": X[i][:, 0]})
    
        # Reclustering the jets
        reclustered_jets, inclusive_jet = reclusterJets(particleAwk, R=0.4, pt_cut=0)
    
        # For jet kinematics plot
        inclusive_jets[i][1] = ak.to_numpy(ak.unzip(inclusive_jet))[0]
        inclusive_jets[i][2] = ak.to_numpy(ak.unzip(inclusive_jet))[1]
        inclusive_jets[i][3] = ak.to_numpy(ak.unzip(inclusive_jet))[2]
        inclusive_jets[i][0] = ak.to_numpy(ak.unzip(inclusive_jet))[3]

        # For particle graphs 
        X[i][:, 1] = ak.to_numpy(ak.unzip(reclustered_jets))[0]
        X[i][:, 2] = ak.to_numpy(ak.unzip(reclustered_jets))[1]
        X[i][:, 3] = ak.to_numpy(ak.unzip(reclustered_jets))[2]
        X[i][:, 0] = ak.to_numpy(ak.unzip(reclustered_jets))[3]
    
        X[i] = energyflow.ptyphims_from_p4s(X[i])
        
        # deleting the mass column from the jets
        X[i] = np.delete(X[i], 3, 1)

    plot_jet_kinematics(inclusive_jets)
    exit()
    
    print("  Reclustering done.")

    # Preprocess by centering jets and normalizing pts
    for x in tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X)):
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()

    # Calculate EnergyEnergyCorrelation (EEC) features
    if additional_edge_attrs == 'eec_with_charges':
        print(f'  Calculating EEC features with charges...')
        additional_edge_attrs = []
        for i in range(len(eec_prop[0])):
            additional_edge_attrs.append(get_eec_ls_values(X, N=eec_prop[0][i], bins=eec_prop[1], axis_range=eec_prop[2]))

    if additional_edge_attrs == 'eec_without_charges':
        print(f'  Calculating EEC features without charges...')
        additional_edge_attrs = []
        for i in range(len(eec_prop[0])):
            additional_edge_attrs.append(get_eec_ls_values(X, N=eec_prop[0][i], bins=eec_prop[1], axis_range=eec_prop[2]))

    # Create output directory if it doesn't exist
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

    # Construct graph as PyG data object
    graph_label = torch.tensor(label, dtype=torch.bool)


    # Add additional attributes if provided
    # if additional_node_attrs:
    #     graph.node_attrs = torch.tensor(additional_node_attrs, dtype=torch.float)

    if additional_edge_attrs:
        # Calculate edge features
        edge_features = [[] for _ in range(len(additional_edge_attrs))]
        for i, j in zip(row, col):
            delta_y = x[i][1] - x[j][1]
            delta_phi_abs = abs(x[i][2] - x[j][2])
            delta_phi = delta_phi_abs if delta_phi_abs <= np.pi else 2 * np.pi - delta_phi_abs
            delta_R = np.sqrt(delta_y**2 + delta_phi**2)
            
            # if delta_R > 0.7:
            #     print(f"delta_R: {delta_R}, delta_y: {delta_y}, delta_phi: {delta_phi}")

            # Determine the bin for the edge value
            bin_index = np.digitize(delta_R, bins=additional_edge_attrs[0].bin_edges()) - 1

            # print(additional_edge_attrs.bin_edges(), bin_index, delta_R, additional_edge_attrs.get_hist_errs(0, False)[0])
            # exit()

            # Get the histogram value for the bin
            for i in range(len(additional_edge_attrs)):
                edge_features[i].append(additional_edge_attrs[i].get_hist_errs(0, False)[0][bin_index])


        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        edge_features_tensor = edge_features_tensor.t() # Transpose to dim (n_edges, n_features)

        # # Zero pad edge_features_tensor to match the dimension of edge_indices_long
        # if edge_features_tensor.size(0) < edge_indices_long.size(1):
        #     padding_size = edge_indices_long.size(1) - edge_features_tensor.size(0)
        #     edge_features_tensor = torch.nn.functional.pad(edge_features_tensor, (0, 0, 0, padding_size))

        # print(edge_features_tensor, edge_indices_long, edge_features_tensor.size(), edge_indices_long.size())
        # exit()

        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=edge_features_tensor, y=graph_label)

    # if additional_graph_attrs:
    #     graph.graph_attrs = torch.tensor(additional_graph_attrs, dtype=torch.float)

    # if additional_hypergraph_attrs:
    #     graph.hypergraph_attrs = torch.tensor(additional_hypergraph_attrs, dtype=torch.float)

    else:
        graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=None, y=graph_label)

    return graph

#_construct_particle_graphs_pyg("./graph_objects/particle_graphs/.", ['fully_connected'], 100000)

_construct_particle_graphs_pyg("./graph_objects/particle_graphs/.", ['fully_connected'], 1500, additional_edge_attrs='eec_with_charges')
