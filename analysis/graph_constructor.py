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
import awkward as ak
import matplotlib.pyplot as plt

from import_dataset import import_CMS2011AJets_dataset
from utils import get_eec_ls_values, plot_jet_kinematics, reclusterJets, OneHotEncodeType, normalize_array

from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinear

def _construct_particle_graphs_pyg(
        output_dir,
        graph_structures,
        N=500000,
        dataset='jetnet',
        recluster_jets=False,
        eec_prop=[[2], 500, (1e-3, 2)], # [N, bins, (R_Lmin, R_Lmax)]
        additional_node_attrs=None,
        additional_edge_attrs=None, # None or eec_with_pids or eec_with_charges or eec_without_charges
        additional_graph_attrs=None,
        additional_hypergraph_attrs=None,
        data_args_jetnet = {
            "jet_type": ["q", "g"],  # gluon and top quark jets
            "data_dir": "datasets/jetnet",
            # these are the default particle features, written here to be explicit
            "particle_features": ["ptrel", "etarel", "phirel", "mask"],
            "num_particles": 30,  # we retain only the 30 highest pT particles for this demo
            "jet_features": ["type", "pt", "eta", "mass"],
            # "particle_normalisation": FeaturewiseLinear(
            #     normal=True, normalise_features=[False, False, False]
            # ),
            # pass our function as a transform to be applied to the jet features
            # "jet_transform": OneHotEncodeType,
            "download": True,
        }
):
    '''
    Construct a list of PyG graphs for the particle-based GNNs, loading from the energyflow dataset

    Graph structure:
        - Nodes: particle four-vectors
        - Edges: no edge features
        - Connectivity: fully connected 
    '''
    
    # BUG: energyflow dataset has a bug 
    if dataset == 'energyflow':
        print(f'Constructing PyG particle graphs from energyflow dataset...')

        # Load dataset
        X, y = energyflow.qg_jets.load(N, pad=False)
        # energyflow.utils.remap_pids(X)

        # Making an empty list to store the old features
        old_X = [0 for _ in range(len(X))]

        
        if reclusterJets:
        
            # Reclustering the jets using fastjet to check the clustering
            print(f'  Reclustering jets using fastjet...')
            
            inclusive_jets = [[0, 0, 0, 0] for _ in range(len(X))]

            for i in range(len(X)):

                # Removing the zero-padded rows
                X[i] = X[i][~np.all(X[i] == 0, axis=1)]

                # Removing the rows with NaN values
                X[i] = X[i][~np.isnan(X[i]).any(axis=1)]

                # Storing the particle and jet E
                energy_values = [0, 0]

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

                # Storing the jet kinematics in a list
                inclusive_jets[i] = [arr[0] for arr in inclusive_jets[i]]

                # Storing the jet E
                energy_values[1] = inclusive_jets[i][0]

                # For particle graphs
                X[i][:, 1] = ak.to_numpy(ak.unzip(reclustered_jets))[0]
                X[i][:, 2] = ak.to_numpy(ak.unzip(reclustered_jets))[1]
                X[i][:, 3] = ak.to_numpy(ak.unzip(reclustered_jets))[2]
                X[i][:, 0] = ak.to_numpy(ak.unzip(reclustered_jets))[3]

                # Storing the particle E
                energy_values[0] = X[i][:, 0]

                # Retrieving the particle coordinates in hadronic coordinates (pt, y, phi)
                X[i] = energyflow.ptyphims_from_p4s(X[i], mass=False)

                #Retrieving the Jet coordinates in hadronic coordinates (pt, y, phi)
                inclusive_jets[i] = energyflow.ptyphims_from_p4s(inclusive_jets[i], mass=False)


                # Making the new features according to the jet tagging papers
                old_X[i] = X[i] # Storing the old features of X (particles)
                old_X[i] = np.array(old_X[i])

                X[i] = np.zeros((len(X[i]), 7)) # Making a new array to store the new features

                # Storing the old features in the new array
                X[i][:, 0] = old_X[i][:, 1] - inclusive_jets[i][1] # delta_y
                X[i][:, 1] = old_X[i][:, 2] - inclusive_jets[i][2] # delta_phi
                X[i][:, 2] = np.log(old_X[i][:, 0]) # log(pt)
                X[i][:, 3] = np.log(energy_values[0]) # particle E
                X[i][:, 4] = np.log(old_X[i][:, 0] / inclusive_jets[i][0]) # log(pt / jet pt)
                X[i][:, 5] = np.log(energy_values[0] / energy_values[1]) # log(E / jet E)
                X[i][:, 6] = np.sqrt(X[i][:, 0]**2 + X[i][:, 1]**2) # delta_R

                # deleting the mass column from the jets
                # X[i] = np.delete(X[i], 3, 1)

            # plot_jet_kinematics(inclusive_jets)
            
            print("  Reclustering done.")


        # Preprocess by normalizing features
        for i, x in enumerate(tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X))):
            mask = ~np.isclose(x[:, 0], 0)
            
            # Apply the mask to eliminate rows with 0 values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with 0 values in old_X[i]
            old_X[i] = old_X[i][mask]

            # Create a mask to identify rows without NaN values in X[i]
            mask = ~np.isnan(X[i]).any(axis=1)

            # Apply the mask to eliminate rows with NaN values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with NaN values in old_X[i]
            old_X[i] = old_X[i][mask]

            # normalizing new features
            X[i] = (X[i] - np.average(X[i], axis=0)) / np.std(X[i], axis=0)



    if dataset == 'jetnet':
        print(f'Constructing PyG particle graphs from JetNet dataset...')

        # Load dataset
        X, y = JetNet(**data_args_jetnet)[:]
        X = X.numpy()
        y = y.numpy()#[:, 0].astype(int)


        # Reshaping and filtering out zero-padded rows
        result = []
        
        for i in range(X.shape[0]):
            # Filter out zero-padded rows
            non_zero_particles = X[i][~np.all(X[i] == 0, axis=1)]
            result.append(non_zero_particles)

        X = result

        # Making an empty list to store the old features
        old_X = [0 for _ in range(len(X))]

        for i in range(len(X)):

            # Making the new features according to the jet tagging papers

            # Storing the old features of X (particles)
            old_X[i] = np.array(X[i][:, :3])
            mask = X[i][:, 3].astype(bool)
            old_X[i] = old_X[i][mask]

            # in the form of particle pt, rel_y, rel_phi
            old_X[i][:, 0] = old_X[i][:, 0] * y[i][1] # pt

            X[i] = np.zeros((len(X[i][mask]), 7)) # Making a new array to store the new features

            # Storing the old features in the new array
            X[i][:, 0] = old_X[i][:, 1] # delta_y
            X[i][:, 1] = old_X[i][:, 2] # delta_phi
            X[i][:, 2] = np.log(old_X[i][:, 0]) # log(pt)
            X[i][:, 3] = np.log(old_X[i][:, 0] * np.cosh(old_X[i][:, 1] + y[i][2])) # log(particle E)
            X[i][:, 4] = np.log(old_X[i][:, 0] / y[i][1]) # log(pt / jet pt)
            X[i][:, 5] = X[i][:, 3] - np.log(np.sqrt(y[i][1]**2 + y[i][3]**2)) # log(E / jet E)
            X[i][:, 6] = np.sqrt(X[i][:, 0]**2 + X[i][:, 1]**2) # delta_R

            X[i] = np.array(X[i])
            old_X[i] = np.array(old_X[i])


        # Preprocess by normalizing features
        for i, x in enumerate(tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X))):
            mask = ~np.isclose(x[:, 0], 0)
            
            # Apply the mask to eliminate rows with 0 values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with 0 values in old_X[i]
            old_X[i] = old_X[i][mask]

            # Create a mask to identify rows without NaN values in X[i]
            mask = ~np.isnan(X[i]).any(axis=1)

            # Apply the mask to eliminate rows with NaN values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with NaN values in old_X[i]
            old_X[i] = old_X[i][mask]

            # normalizing new features
            X[i] = (X[i] - np.average(X[i], axis=0)) / np.std(X[i], axis=0)

        

        # One-hot encode the labels
        y = OneHotEncodeType(y)[:, :1]

        # plot_jet_kinematics(X, input_type='hadronic')
        
        # plotting
        fig, axs = plt.subplots(1, 7, figsize=(20, 5))
        
        jet_list = [np.array([]) for _ in range(7)]
        for i in range(7):
            for j in range(len(X)):
                jet_list[i] = np.append(jet_list[i], X[j][:, i])

        for i in range(7):
            axs[i].hist(jet_list[i], bins=100)
            axs[i].set_title(f'Histogram of X[:, :, {i}]')

        plt.tight_layout()
        plt.savefig('scatter_plot.png')
        exit()

    # Calculate EnergyEnergyCorrelation (EEC) features
    if additional_edge_attrs == 'eec_with_charges':
        print(f'  Calculating EEC features with charges...')
        additional_edge_attrs = []
        for i in range(len(eec_prop[0])):
            additional_edge_attrs.append(get_eec_ls_values(old_X, N=eec_prop[0][i], bins=eec_prop[1], axis_range=eec_prop[2]))

    if additional_edge_attrs == 'eec_without_charges':
        print(f'  Calculating EEC features without charges...')
        additional_edge_attrs = []
        for i in range(len(eec_prop[0])):
            additional_edge_attrs.append(get_eec_ls_values(old_X, N=eec_prop[0][i], bins=eec_prop[1], axis_range=eec_prop[2]))


    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    for graph_structure in graph_structures:
        graph_key = f'particle__{graph_structure}'

        graph_list = []
        args = [(x, y[i], old_X[i]) for i, x in enumerate(X)] # Using old_X to store the old features
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
    x, label, old_x = args

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
        edge_features = [[] for _ in range(len(additional_edge_attrs) + 4)]   # 4 additional features: delta, k_T, z, m^2
        
        # Normalizing the eec values
        edge_attrs = []

        for i in range(len(additional_edge_attrs)):
            edge_attrs.append(normalize_array(np.array(list(additional_edge_attrs[i].get_hist_errs(0, False)[0]))))

        for i, j in zip(row, col):

            # Calculating delta_R to for EEC
            delta_y = old_x[i][1] - old_x[j][1]  # old_x is used since new features were created in x
            delta_phi_abs = abs(old_x[i][2] - old_x[j][2])   # old_x is used since new features were created in x
            delta_phi = delta_phi_abs if delta_phi_abs <= np.pi else 2 * np.pi - delta_phi_abs
            delta_R = np.sqrt(delta_y**2 + delta_phi**2)
            
            # if delta_R > 1.8:
            #     print(f"delta_R: {delta_R}, delta_y: {delta_y}, delta_phi: {delta_phi}")

            # Determine the bin for the edge value
            bin_index = np.digitize(delta_R, bins=additional_edge_attrs[0].bin_edges()) - 1

            # Calculate other log-transformed edge features
            # Δ (delta)
            delta = delta_R
            
            # k_T
            k_T = min(old_x[i][0], old_x[j][0]) * delta
            
            # z
            z = min(old_x[i][0], old_x[j][0]) / (old_x[i][0] + old_x[j][0])
            
            # m^2 (invariant mass squared)
            # calculate energy components
            E_a = old_x[i][0] * np.cosh(old_x[i][1])
            E_b = old_x[j][0] * np.cosh(old_x[j][1])
            
            # Calculate momentum components
            p_x_a = old_x[i][0] * np.cos(old_x[i][2])
            p_y_a = old_x[i][0] * np.sin(old_x[i][2])
            p_z_a = old_x[i][0] * np.sinh(old_x[i][1])
            
            p_x_b = old_x[j][0] * np.cos(old_x[j][2])
            p_y_b = old_x[j][0] * np.sin(old_x[j][2])
            p_z_b = old_x[j][0] * np.sinh(old_x[j][1])
            
            # Calculate invariant mass squared m^2
            m2 = (E_a + E_b)**2 - ((p_x_a + p_x_b)**2 + (p_y_a + p_y_b)**2 + (p_z_a + p_z_b)**2)
            
            # Log-transformed features
            ln_delta = np.log(delta)
            ln_k_T = np.log(k_T)
            ln_z = np.log(z)
            ln_m2 = np.log(m2) if m2 > 0 else 0 # Avoid negative values

            # Get the histogram value for the bin
            for k in range(len(additional_edge_attrs) + 4):
                # EEC values
                if k < len(additional_edge_attrs):
                    edge_features[k].append(edge_attrs[k][bin_index])

                # Additional features
                elif k == len(additional_edge_attrs):
                    edge_features[k].append(ln_delta)
                elif k == len(additional_edge_attrs) + 1:
                    edge_features[k].append(ln_k_T)
                elif k == len(additional_edge_attrs) + 2:
                    edge_features[k].append(ln_z)
                elif k == len(additional_edge_attrs) + 3:
                    edge_features[k].append(ln_m2)


        # Convert edge features to tensor
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        edge_features_tensor = edge_features_tensor.t() # Transpose to dim (n_edges, n_features)

        # Normalize the feature values column-wise from 1 to the end
        means = edge_features_tensor[:, 1:].mean(dim=0)
        stds = edge_features_tensor[:, 1:].std(dim=0)
        edge_features_tensor[:, 1:] = (edge_features_tensor[:, 1:] - means) / stds


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

_construct_particle_graphs_pyg("./graph_objects/particle_graphs/.", ['fully_connected'], 200000, dataset='jetnet', recluster_jets=True, additional_edge_attrs='eec_without_charges')
