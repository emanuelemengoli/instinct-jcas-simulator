from simulation_files.utils.helper_functions import *
from simulation_files.simulator_backbone.plotting_functions import *

class QUEUE:
    def __init__(self) -> None:
        self.shannon_rates = []
        self.wloads = []
        self.arrival_rates = [] #this should be vc_dipendt
        self.trace = {}

    def update_workload(self, arrival_rate:float , shannon_rate: float, t:int):

        old_load = self.wloads[-1] if len(self.wloads) > 0 else 0
        new_load_ = (poisson(lam=arrival_rate), poisson(lam=shannon_rate))
        cum_load_ = max(0,old_load + new_load_[0] - new_load_[1])
        # cum_load_ = max(0,old_load + poisson(lam=self.arrival_rate) - poisson(lam=shannon_rate))
        self.wloads.append(cum_load_)
        self.shannon_rates.append(shannon_rate)
        self.arrival_rates.append(arrival_rate)
        self.trace[t] = new_load_


class KalmanFilter:
    def __init__(self, id, mu0, Sigma0):
        self.id = id
        self.x = mu0  # Initial state estimate (x0)
        self.Sigma = Sigma0  # Initial covariance estimate (Sigma_0)
        # self.t = 0
        self.stability_states = []
        self.estimated_states = []
        self.covariance_states = []

    def update_estimate(self,A,C,y,V,Z):
        #Here we assume that A,Z,C,V can vary over time

        # Compute the Kalman gain
        S = np.dot(np.dot(C, self.Sigma), C.T) + V
        S_p = np.dot(np.dot(A, self.Sigma), C.T)
        KF_GAIN = - np.dot(S_p, np.linalg.inv(S))

        #shortcut
        proj = A + np.dot(KF_GAIN,C)

        #prediction and estimate update
        self.x = np.dot(proj, self.x) - np.dot(KF_GAIN,y) 
        self.Sigma = np.dot(np.dot(proj, self.Sigma),A.T) + Z
        
        # self.t +=1

        self.stability_states.append(proj)
        self.estimated_states.append(self.x.flatten())
        self.covariance_states.append(self.Sigma)
























#     def simulate_kalman_filter(self,):
        
#         self.kf = KalmanFilter(id = self.id, A=self.A_t[0],Z=self.Z, C=self.C, V=self.V_t[0], mu_0=self.mu_0, Sigma_0=self.sigma_0)
#         self.true_state[0] = self.mu_0.flatten()
#         self.simulate_so_translation()
        
#         # Kalman filter estimates
#         for t in range(self.n_steps):
#             A_curr = self.A_t[t]
#             V_curr = self.V_t[t]
#             estimated_state, uncertainty = 
#             self.estimated_states[t] = estimated_state.flatten() #shape (1,4) #.flatten()
#             self.covariance_states[t] = uncertainty

#         self.true_state = np.array(self.true_state)
#         self.observations = np.array(self.observations)
#         self.estimated_states = np.array(self.estimated_states)
#         self.covariance_states = np.array(self.covariance_states)
    
# class KF_manager():

#     def __init__(self, so, meas_dim, n_steps, bs, thr, params) -> None:
#         self.id = so.id
#         self.n = 1 #len(self.sos_)
#         self.d = len(so.theta) #len(self.sos_[0].theta)
#         self.meas_dim = meas_dim
#         self.so = so
#         self.n_steps = n_steps
#         self.bs = bs
#         self.bs_scaler = np.array([self.bs.position[0],self.bs.position[1],1,1])
#         self._init_kf_operators()
#         self._init_kf_params(**params)
#         self._init_kf_output_buffers()
#         self.plotter = Plotter(kf_manager=self, queue_manager = None, thr=thr)
    
#     def _init_kf_output_buffers(self):
#         # Initialize lists to store the outputs of the simulation
#         #we assume 1 obj
#         self.true_state = np.zeros((self.n_steps + 1, self.d)) #dimension of the parameter vector #otherwise  (self.n, self.n_steps + 1, self.d) #[]         # To store the true states
#         self.observations = np.zeros((self.n_steps, self.meas_dim))#return time delay, doppler shift #[]       # To store the observations
#         self.estimated_states =  np.zeros((self.n_steps, self.d))#[]   # To store the estimated states
#         self.covariance_states = np.zeros((self.n_steps, self.d, self.d)) #np.zeros((n_steps, dim_state))#[]  # To store the state covariance estimates

#     def _init_kf_operators(self,):
#          #dimension of the parameter vector
#         #self.build_triangular_matrix = lambda max_:  np.tril(uniform(low = 0, high = max_, size = (self.d,self.d)), k=0)
#         self.build_triangular_matrix = lambda max_:  np.tril(randint(low = 0, high = max_, size = (self.d,self.d)), k=0)
#         self.shur_stability = lambda matrix: np.fill_diagonal(matrix, uniform(0, 1, size = self.d))
#         self.fill_interference = lambda diagonal, id_matrix: np.fill_diagonal(id_matrix, diagonal)

            
    


