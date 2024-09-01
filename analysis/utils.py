import numpy as np
import eec

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
