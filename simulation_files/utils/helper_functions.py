from simulation_files.utils.packages import *
from simulation_files.network_generator.radio_env import *

coin_toss = lambda: random.random() <= 0.5
merge_vectors = lambda v1,v2 : np.concatenate((v1, v2), axis=None).flatten()
formatter = lambda x, dig: float(f"{x:.{dig}e}")
array_formatter = lambda arr,dig: np.array([formatter(x=i,dig=dig) for i in arr])
dict_formatter = lambda d, dig: {k: formatter(x=v,dig=dig) for k, v in d.items()}
flatten = lambda xss: [x for xs in xss for x in xs]
get_array = lambda x: np.array(x)
get_spectral_radius = lambda matrix: max(abs(eigvals(matrix)))
l1_norm = lambda arr1,arr2: np.sum(np.abs(np.array(arr1) - np.array(arr2)))
l2_norm = lambda arr1,arr2: norm(np.array(arr1) - np.array(arr2))
matrix_norm = lambda matrix, ord: norm(matrix, ord=ord)
apply_lambda = lambda func, lst: list(map(func, lst))
# Using a lambda function to apply the conversion
convert_to_m_per_s = lambda v_kmh: v_kmh / 3.6
# Lambda to convert from m/s to km/h
convert_to_km_per_h = lambda v_mps: v_mps * 3.6
get_temporal_avg = lambda array: np.mean(get_array(array))
# get_temporal_avg = lambda array,steady_state_thr: np.mean(get_array(array[steady_state_thr:]))
get_vc_diam = lambda region: polygon_diameter(Polygon(region)) 
f_vc_diam = lambda region: max(exp(-get_vc_diam(region=region)),0.99)

# Define the normalization function
def min_max_normalizer(values, interval=None):
    """Normalize an array or single value to the range [0, 1] or a given interval."""
    if isinstance(values, (list, tuple, np.ndarray)):
        values = np.array(values)
        if interval is None:
            min_val, max_val = np.min(values), np.max(values)
        else:
            min_val, max_val = interval

        if min_val == max_val:
            return values  # Avoid division by zero
        return (values - min_val) / (max_val - min_val)
    else:
        if interval is None:
            raise ValueError("Interval must be provided for single value normalization.")
        return (values - min(interval)) / (max(interval) - min(interval))


def polygon_diameter(polygon: Polygon) -> float:
    """
    Returns the diameter (maximum distance) of the given Shapely polygon.
    """
    # 1. Get the convex hull of the polygon
    hull = polygon.convex_hull
    
    # 2. Extract the hull's exterior coordinates
    coords = list(hull.exterior.coords)

    # 3. Compute all pairwise distances and track the maximum
    max_dist = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist_ij = dist(coords[i], coords[j])  # or use Point.distance(...)
            if dist_ij > max_dist:
                max_dist = dist_ij

    return max_dist

def get_probability_thr(sample: np.ndarray, thr: float, tail: bool = True) -> float:
    """
    Calculates the probability that all elements in the sample are above or below a threshold.
    
    Args:
        sample (np.ndarray): The input sample array.
        thr (float): The threshold value.
        above (bool): If True, calculates the probability of elements being >= thr.
                      If False, calculates the probability of elements being <= thr.
    
    Returns:
        float: Probability w.r.t the threshold [0,1].
    """
    if tail:
        thr_indc = (sample >= thr).astype(int)
    else:
        thr_indc = (sample <= thr).astype(int)

    p_i = thr_indc.mean()
    #this count the elements above the thr vs the total count, giving an idea the % of the net above or below the thr

    return p_i

def get_joint_probability_thr(
    w_sample: np.ndarray,
    kf_sample: np.ndarray,
    w_thr: float,
    kf_thr: float,
    tail: bool = False,
) -> float:
    """
    Calculates the joint probability that elements in w_sample and kf_sample
    simultaneously meet their respective threshold conditions.
    
    Args:
        w_sample (np.ndarray): The first input sample array.
        kf_sample (np.ndarray): The second input sample array.
        w_thr (float): The threshold for w_sample.
        kf_thr (float): The threshold for kf_sample.
        tail (bool): Condition for w_sample and kf_sample. If True, elements >= _thr.
                        If False, elements <= _thr.
    
    Returns:
        float: The joint probability as the proportion of samples meeting both conditions.
    """
    # 1) Ensure no length mismatch
    if len(w_sample) != len(kf_sample):
        raise ValueError("w_sample and kf_sample must have the same length.")

    # 2) Get indicators for w_sample
    if tail:
        w_indicator = (w_sample >= w_thr).astype(int)
        kf_indicator = (kf_sample >= kf_thr).astype(int)
    else:
        kf_indicator = (kf_sample <= kf_thr).astype(int)
        w_indicator = (w_sample <= w_thr).astype(int)
    
    # 3) Compute joint indicators: both conditions must be met
    joint_indicator = w_indicator * kf_indicator  # 1 if both conditions are met, else 0
    
    # 4) Calculate joint probability as the mean of joint indicators
    p_joint = joint_indicator.mean()

    return p_joint

def get_empirical_distribution(array):

    digits = 2
    array = array_formatter(array, digits)
    #print('steady_state_covs',steady_state_covs)
    #n = len(self.bss)
    n = len(array)
    cn = Counter(array)
    #print('cn',cn)
    _values = get_array(list(cn.keys()))
    _probs = get_array(list(cn.values()))/n

    return _values, _probs

def torus_distance(x, y):
    """
    Computes the wrap-around (toroidal) distance between two points x, y
    in the rectangle [-Wx, Wx] x [-Hy, Hy].
    
    Parameters
    ----------
    x, y : array-like of shape (2,)
        Coordinates (x, y) of the two points. Each x in [-Wx, Wx], y in [-Hy, Hy].
    Wx, Hy : float
        The half-extent in each dimension. The full domain is 2Wx by 2Hy.
    
    Returns
    -------
    dist : float
        The Euclidean distance taking into account wrap-around edges.
    """
    # Convert to numpy arrays
    x, y = np.asarray(x), np.asarray(y)
    
    dx = np.abs(x[0] - y[0])
    dy = np.abs(x[1] - y[1])
    
    # The total "width" and "height" of the toroidal space
    width = 2 * Wx
    height = 2 * Hy
    
    # Compute toroidal distances
    dx = min(dx, width - dx)
    dy = min(dy, height - dy)

    return np.sqrt(dx**2 + dy**2)


def lookup_bs(items: list, id: int):
    """
    General-purpose method to retrieve all objects from a list using their identifier.
    
    Parameters:
    - items (List): List of items to search through.
    - identifier (int): The identifier of the items to retrieve.
    
    Returns:
    - List: List of items that match the identifier, empty list if none are found.
    """
    if items:
        return [obj for obj in items if obj.bs.id == id]
    return []


# def indicator_thr(sample: np.ndarray, thr: float, tail: bool = True) -> float:
#     """
#     Calculates the probability that all elements in the sample are above or below a threshold.
    
#     Args:
#         sample (np.ndarray): The input sample array.
#         thr (float): The threshold value.
#         above (bool): If True, calculates the probability of elements being >= thr.
#                       If False, calculates the probability of elements being <= thr.
    
#     Returns:
#         float: Probability w.r.t the threshold [0,1].
#     """
#     if tail:
#         thr_indc = (sample >= thr).astype(int)
#     else:
#         thr_indc = (sample <= thr).astype(int)

#     return thr_indc

    # def bs_lookup(self, bs_id: int = None) -> Optional[BS]:
    #     """
    #     Retrieves a BS object from the list of BSs using its identifier.
        
    #     Parameters:
    #     - bs_id (int): The identifier of the BS to retrieve.
        
    #     Returns:
    #     - Optional[BS]: The BS object if found, None otherwise.
    #     """
    #     return self._lookup(self.BSs, bs_id) if bs_id is not None else None
    
    # def ue_lookup(self, ue_id: int = None) -> Optional[UE]:
    #     """
    #     Retrieves a UE object from the list of UEs using its identifier.
        
    #     Parameters:
    #     - ue_id (int): The identifier of the UE to retrieve.
        
    #     Returns:
    #     - Optional[UE]: The UE object if found, None otherwise.
    #     """
    #     return self._lookup(self.UEs, ue_id) if ue_id is not None else None




def scale_matrix_to_stable(matrix, scaler):
    spectral_radius = get_spectral_radius(matrix)
    print('spectral radius', spectral_radius)
    # Scale the matrix to have largest eigenvalue < 1
    if spectral_radius >= 1:
        matrix = matrix / (spectral_radius)
        matrix *= scaler 
        # print(matrix)
        # print(get_spectral_radius(matrix))
        # print(matrix_norm(matrix))
    return matrix

def add_noise_to_matrix(matrix):
    rows, cols = matrix.shape
    noise = normal(loc=0, scale=1, size=(rows, cols))
    return matrix + noise

def generate_random_matrix(rows, cols, distribution='normal', mean=0, std_dev=1):
    """
    Generates a random matrix of specified dimensions.
    
    Parameters:
    - rows: Number of rows in the matrix.
    - cols: Number of columns in the matrix.
    - distribution: Type of distribution to use ('normal' or 'uniform').
    - mean: Mean value for the normal distribution (default is 0).
    - std_dev: Standard deviation for the normal distribution (default is 1).
    
    Returns:
    - A random matrix of shape (rows, cols).
    """
    mx = np.eye(N=rows, M = cols)

    if distribution == 'normal':
        mx+= normal(loc=mean, scale=std_dev, size=(rows, cols))

    elif distribution == 'uniform':
        lb = mean-3*std_dev
        ub = mean+3*std_dev
        mx+= uniform(low=lb, high=ub, size=(rows, cols))
    
    else:
        raise ValueError("Unsupported distribution type. Use 'normal' or 'uniform'.")

    return mx