import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import eec
import fastjet
from sklearn.preprocessing import OneHotEncoder

from joblib import Parallel, delayed, Memory
import numba as nb

# Set up shared memory for large arrays
memory = Memory(location='/tmp/joblib_cache', verbose=0)

# Precompute RL_n for a hyperedge using Numba
@nb.njit(fastmath=True)
def compute_RL_n(coords):
    """
    Compute RL_n (maximum pairwise angular distance) for a given hyperedge.
    
    Args:
        coords (np.ndarray): Array of shape (n, 2), where each row contains [delta_y, delta_phi].
    
    Returns:
        float: Maximum pairwise angular distance RL_n.
    """
    n = coords.shape[0]
    max_distance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy = coords[i, 0] - coords[j, 0]
            dphi = abs(coords[i, 1] - coords[j, 1])
            if dphi > np.pi:
                dphi = 2 * np.pi - dphi
            distance = np.sqrt(dy**2 + dphi**2)
            if distance > max_distance:
                max_distance = distance
    return max_distance

# Process a single hyperedge using Numba-compatible operations
def process_hyperedge(hyperedge, old_x, precomputed_bin_edges, precomputed_hist_values, eec2):
    """
    Process a single hyperedge to compute RL_n and EEC values.

    Args:
        hyperedge (tuple): Indices of nodes in the hyperedge.
        old_x (np.ndarray): Node feature matrix of shape [num_nodes, d].
        precomputed_bin_edges (list): Precomputed bin edges for all histograms.
        precomputed_hist_values (list): Precomputed histogram values for all histograms.
        eec2 (np.ndarray): Precomputed 2-point EEC values.

    Returns:
        tuple: Hyperedge and its computed EEC values.
    """
    # Extract coordinates for the nodes in this hyperedge
    coords = old_x[list(hyperedge), 1:3]  # Extract [delta_y, delta_phi]
    
    # Compute RL_n using the optimized Numba function
    RL_n = compute_RL_n(coords)
    
    # Compute EEC values for this hyperedge
    eec_vals = []
    for bin_edges, hist_values in zip(precomputed_bin_edges, precomputed_hist_values):
        bin_index = np.digitize(RL_n, bins=bin_edges) - 1
        hist_values_normalized = np.divide(hist_values, eec2,
                                           out=np.zeros_like(hist_values),
                                           where=eec2 != 0)
        norm_hist_values = normalize_array(hist_values_normalized)
        
        if 0 <= bin_index < len(norm_hist_values):
            eec_vals.append(norm_hist_values[bin_index])
        else:
            eec_vals.append(0.0)
    
    return hyperedge, eec_vals

def process_hyperedge_batch(hyperedge_batch, old_x_shared, precomputed_bin_edges,
                            precomputed_hist_values, eec2_shared):
    """
    Process a batch of hyperedges.

    Args:
        hyperedge_batch (list): List of hyperedges to process.
        old_x_shared (np.ndarray): Shared memory-mapped node feature matrix.
        precomputed_bin_edges (list): Precomputed bin edges for all histograms.
        precomputed_hist_values (list): Precomputed histogram values for all histograms.
        eec2_shared (np.ndarray): Shared memory-mapped 2-point EEC values.

    Returns:
        list: List of processed hyperedges and their attributes.
    """
    results = []
    for hyperedge in hyperedge_batch:
        results.append(process_hyperedge(hyperedge, old_x_shared,
                                         precomputed_bin_edges,
                                         precomputed_hist_values,
                                         eec2_shared))
    return results

def construct_n_point_hyperedges(num_nodes, old_x, additional_hypergraph_attrs, n, eec2):
    """
    Constructs hyperedges for n-point energy-energy correlators computing their EEC values.

    Args:
      - num_nodes: Number of nodes.
      - old_x: Numpy array of node features [num_nodes, d] with at least [pt, delta_y, delta_phi].
      - additional_hypergraph_attrs: List of histogram objects.
      - n: Either an integer or a list of integers.
      - eec2: Numpy array representing 2-point EEC values.

    Returns:
      - hyperedge_index: Torch sparse COO tensor (num_nodes, M).
      - hyperedge_attr: Torch tensor (M, F) with EEC values.
    """
    # Ensure n is a list
    n_list = [n] if isinstance(n, int) else n

    # Memory-map large arrays to avoid serialization overhead
    old_x_shared = memory.cache(np.array)(old_x)
    eec2_shared = memory.cache(np.array)(eec2)

    # Precompute static data from additional_hypergraph_attrs
    precomputed_bin_edges = [hist_obj.bin_edges() for hist_obj in additional_hypergraph_attrs]
    precomputed_hist_values = [np.array(list(hist_obj.get_hist_errs(0, False)[0])) 
                                for hist_obj in additional_hypergraph_attrs]

    all_hyperedges = []
    all_hyperedge_attrs = []

    # Loop over each n-value and iterate directly using the iterator
    for n_val in n_list:
        if n_val > num_nodes:
            continue
        
        # Generate combinations lazily using itertools.combinations
        hyperedges = list(itertools.combinations(range(num_nodes), n_val))
        
        # Chunk hyperedges into batches
        batch_size = 2500
        hyperedge_batches = [hyperedges[i:i + batch_size] 
                             for i in range(0, len(hyperedges), batch_size)]
        
        # Process batches in parallel using Joblib
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_hyperedge_batch)(batch,
                                             old_x_shared,
                                             precomputed_bin_edges,
                                             precomputed_hist_values,
                                             eec2_shared) 
            for batch in hyperedge_batches
        )

        print(f"Finished processing {n_val}-point hyperedges.")

        # Flatten results from parallel execution
        for batch_result in results:
            for processed_hyperedge, eec_vals in batch_result:
                all_hyperedges.append(processed_hyperedge)
                all_hyperedge_attrs.append(eec_vals)

    total_hyperedges = len(all_hyperedges)
    
    # Build the sparse incidence matrix.
    row_indices = []
    col_indices = []
    for hyperedge_id, hyperedge in enumerate(all_hyperedges):
        for node in hyperedge:
            row_indices.append(node)
            col_indices.append(hyperedge_id)
    
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    
    # Create a sparse COO tensor for the incidence matrix
    hyperedge_index = torch.sparse_coo_tensor(indices, values,
                                              size=(num_nodes, total_hyperedges))
    
    # Convert the list of hyperedge attributes to a tensor
    hyperedge_attr = torch.tensor(all_hyperedge_attrs, dtype=torch.float32)
    
    return hyperedge_index, hyperedge_attr


def get_eec_ls_values(data, N = 2, bins = 50, axis_range = (1e-3, 1)):
    """
    Get the EEC values for the given data.
    
    Parameters:
    data: np.ndarray
        The data for which the EEC values are to be calculated.
    N: int
        The number of nearest neighbors to consider.
    bins: int
        The number of bins to use for the histogram.
    axis_range: tuple
        The range of the x-axis.
        
    Returns:
    eec_ls: The EEC histogram with the bins and the values.
        The EEC values.
    """

    # Get the EEC values
    # Create an instance of the EECLongestSide class
    eec_ls = eec.EECLongestSideId(N, bins, axis_range)

    # Multicore compute for EECLongestSide
    eec_ls(data)
    print(eec_ls)

    # Scaling eec values
    eec_ls.scale(1/eec_ls.sum())

    return eec_ls

# function to one hot encode the jet type and leave the rest of the features as is
def OneHotEncodeType(x: np.ndarray):
    enc = OneHotEncoder(categories=[[0, 1]])
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, 3)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)

@nb.njit(fastmath=True)
def normalize_array(arr):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    normalized_arr = (arr - mean) / std_dev
    return normalized_arr


def reclusterJets(jet, R=0.4, pt_cut=0):
    """
    Recluster the jets.
    
    Parameters:
    jet: np.ndarray
        The jets to be reclusted.
    R: float
        The radius parameter.
    pt_cut: float
        The pt cut.
        
    Returns:
    reclustered_jets: np.ndarray
        The reclustered jets.
    """

    # Create a jet definition
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)

    # Create a cluster sequence
    cs = fastjet.ClusterSequence(jet, jet_def)

    return cs.constituents()[0], cs.inclusive_jets()

def plot_jet_kinematics(inclusive_jet, input_type=''):
    pt_list = []
    y_list = []
    phi_list = []
    
    for jet in inclusive_jet:

        if input_type=='hadronic':

            pt = jet[:, 0]
            y = jet[:, 1]
            phi = jet[:, 2]

            print(pt)
            # pt = jet[0]
            # y = jet[1]
            # phi = jet[2]

        else:
            # Extract E, px, py, pz
            E = jet[0]
            px = jet[1]
            py = jet[2]
            pz = jet[3]
            
            # Calculate pt, y, phi
            pt = np.sqrt(px**2 + py**2)
            y = 0.5 * np.log((E + pz) / (E - pz))
            phi = np.arctan2(py, px)
        
        # Append to lists
        pt_list.append(pt)
        y_list.append(y)
        phi_list.append(phi)
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot pt 
    axs[0].hist(pt_list, bins=50)
    axs[0].set_xlabel('pt')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('pt')

    # Plot y
    axs[1].hist(y_list, bins=50)
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('y')
    
    # Plot phi
    axs[2].hist(phi_list, bins=50)
    axs[2].set_xlabel('phi')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('phi')


    # # Plot pt vs y
    # axs[0].scatter(pt_list, y_list)
    # axs[0].set_xlabel('pt')
    # axs[0].set_ylabel('y')
    # axs[0].set_title('pt vs y')
    # 
    # # Plot y vs phi
    # axs[1].scatter(y_list, phi_list)
    # axs[1].set_xlabel('y')
    # axs[1].set_ylabel('phi')
    # axs[1].set_title('y vs phi')
    #
    # # Plot pt vs phi
    # axs[2].scatter(pt_list, phi_list)
    # axs[2].set_xlabel('pt')
    # axs[2].set_ylabel('phi')
    # axs[2].set_title('pt vs phi')

    # Save the plot
    plt.tight_layout()
    plt.savefig('kinematics_plot.png')
    

# BUG: The following function is not working as expected
def ms2pids(ms):
    """
    Convert the masses to pids.
    
    Parameters:
    ms: np.ndarray
        The masses to convert.
        
    Returns:
    pids: np.ndarray
        The pids.
    """

    pidsDict = {
    #   PDGID     CHARGE MASS          NAME
        0:       ( 0.,   0.,      ), # void
        1:       (-1./3, 0.33,    ), # down
        2:       ( 2./3, 0.33,    ), # up
        3:       (-1./3, 0.50,    ), # strange
        4:       ( 2./3, 1.50,    ), # charm
        5:       (-1./3, 4.80,    ), # bottom
        6:       ( 2./3, 171.,    ), # top
        11:      (-1.,   5.11e-4, ), # e-
        12:      ( 0.,   0.,      ), # nu_e
        13:      (-1.,   0.10566, ), # mu-
        14:      ( 0.,   0.,      ), # nu_mu
        15:      (-1.,   1.77682, ), # tau-
        16:      ( 0.,   0.,      ), # nu_tau
        21:      ( 0.,   0.,      ), # gluon
        22:      ( 0.,   0.,      ), # photon
        23:      ( 0.,   91.1876, ), # Z
        24:      ( 1.,   80.385,  ), # W+
        25:      ( 0.,   125.,    ), # Higgs
        111:     ( 0.,   0.13498, ), # pi0
        113:     ( 0.,   0.77549, ), # rho0
        130:     ( 0.,   0.49761, ), # K0-long
        211:     ( 1.,   0.13957, ), # pi+
        213:     ( 1.,   0.77549, ), # rho+
        221:     ( 0.,   0.54785, ), # eta
        223:     ( 0.,   0.78265, ), # omega
        310:     ( 0.,   0.49761, ), # K0-short
        321:     ( 1.,   0.49368, ), # K+
        331:     ( 0.,   0.95778, ), # eta'
        333:     ( 0.,   1.01946, ), # phi
        445:     ( 0.,   3.55620, ), # chi_2c
        555:     ( 0.,   9.91220, ), # chi_2b
        2101:    ( 1./3, 0.57933, ), # ud_0
        2112:    ( 0.,   0.93957, ), # neutron
        2203:    ( 4./3, 0.77133, ), # uu_1
        2212:    ( 1.,   0.93827, ), # proton
        1114:    (-1.,   1.232,   ), # Delta-
        2114:    ( 0.,   1.232,   ), # Delta0
        2214:    ( 1.,   1.232,   ), # Delta+
        2224:    ( 2.,   1.232,   ), # Delta++
        3122:    ( 0.,   1.11568, ), # Lambda0
        3222:    ( 1.,   1.18937, ), # Sigma+
        3212:    ( 0.,   1.19264, ), # Sigma0
        3112:    (-1.,   1.19745, ), # Sigma-
        3312:    (-1.,   1.32171, ), # Xi-
        3322:    ( 0.,   1.31486, ), # Xi0
        3334:    (-1.,   1.67245, ), # Omega-
        10441:   ( 0.,   3.41475, ), # chi_0c
        10551:   ( 0.,   9.85940, ), # chi_0b
        20443:   ( 0.,   3.51066, ), # chi_1c
        9940003: ( 0.,   3.29692, ), # J/psi[3S1(8)]
        9940005: ( 0.,   3.75620, ), # chi_2c[3S1(8)]
        9940011: ( 0.,   3.61475, ), # chi_0c[3S1(8)]
        9940023: ( 0.,   3.71066, ), # chi_1c[3S1(8)]
        9940103: ( 0.,   3.88611, ), # psi(2S)[3S1(8)]
        9941003: ( 0.,   3.29692, ), # J/psi[1S0(8)]
        9942003: ( 0.,   3.29692, ), # J/psi[3PJ(8)]
        9942033: ( 0.,   3.97315, ), # psi(3770)[3PJ(8)]
        9950203: ( 0.,   10.5552, ), # Upsilon(3S)[3S1(8)]
    }

    particleMassesDict  = {pdgid: props[1] for pdgid,props in pidsDict.items()}

    pids = []
    for m in ms:
        for pdgid, mass in particleMassesDict.items():
            print(mass, m)
            if np.isclose(mass, m):
                pids.append(pdgid)

    pids = np.array(pids)

    return pids
