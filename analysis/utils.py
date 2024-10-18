import numpy as np
import matplotlib.pyplot as plt
import eec
import fastjet

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

def plot_jet_kinematics(inclusive_jet):
    pt_list = []
    y_list = []
    phi_list = []
    
    for jet in inclusive_jet:
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
    
    # Plot pt vs y
    axs[0].scatter(pt_list, y_list)
    axs[0].set_xlabel('pt')
    axs[0].set_ylabel('y')
    axs[0].set_title('pt vs y')
    
    # Plot y vs phi
   axs[1].scatter(y_list, phi_list)
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('phi')
    axs[1].set_title('y vs phi')
    
    # Plot pt vs phi
    axs[2].scatter(pt_list, phi_list)
    axs[2].set_xlabel('pt')
    axs[2].set_ylabel('phi')
    axs[2].set_title('pt vs phi')
    
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
