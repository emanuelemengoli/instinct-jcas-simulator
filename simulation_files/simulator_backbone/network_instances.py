# from utils_and_libs import *
from simulation_files.simulator_backbone.kf_queue import *
from simulation_files.network_generator.radio_env import *
import math
# import itertools


def get_C_process(parameter, dim):
    return uniform(low=-parameter, high=parameter, size=dim)

def get_A_process(parameter):
    return np.array(
                [
                [1,0, 1, 0],  
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ]
                ) * parameter 

def get_Z_process(parameter, dim):
    return np.identity(n=dim, dtype=float) * parameter

def f_interf(interference, dim, signal_power):
    return interference / (dim * signal_power)

def cap_sinr(sinr):
    return min(sinr, 256)


def get_V_process(args): #to add signal stength and 
    interference = args['interference']
    dim = args['dim']
    signal_power = args['signal_power']
    return np.identity(n=dim, dtype=float) * f_interf(interference=interference, dim=dim, signal_power=signal_power)

# class DirectionalBeamformer:
#     def __init__(self, main_lobe_gain, side_lobe_gain, beams):
#         self.main_lobe_gain = main_lobe_gain
#         self.side_lobe_gain = side_lobe_gain
#         self.n_beams = 2**(beams)
#         self.beamwidth = (2*math.pi)/self.n_beams
#         #returns the angle where the BS is poising, it is an iterator
#         #reference_vector = np.array([1, 0])  # Assume is initialized with BS faces x-direction
#         self.reference_vector_iter = (
#             lambda: (np.array([math.cos(i * self.beamwidth/2), math.sin(i * self.self.beamwidth/2)])
#                     for i in itertools.count())
#         )()
#         self.reference_vector = next(self.reference_vector_iter)
    
#     def realtive_angle(self, v1, v2):
#         v1_u = v1 / norm(v1)
#         v2_u = v2 / norm(v2)
#         angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#         return angle

#     def move_reference_vector(self):
#         self.reference_vector = next(self.reference_vector_iter)

#     def beamforming_gain(self, target_pos,reference_pos):

#         direction_vector = np.array(target_pos) - np.array(reference_pos)
        
#         angle = self.realtive_angle(direction_vector, self.reference_vector)
#         if angle <= self.beamwidth / 2:
#             return self.main_lobe_gain
#         else:
#             return self.side_lobe_gain

class DirectionalBeamformer:
    def __init__(self, main_lobe_gain: float, side_lobe_gain: float, log2_beams: int):
        self.main_lobe_gain = main_lobe_gain
        self.side_lobe_gain = side_lobe_gain
        self.n_beams = 2 ** log2_beams
        self.beamwidth = (2 * math.pi) / self.n_beams

        # Precompute beam direction vectors (unit vectors)
        self.beam_vectors = [
            np.array([math.cos(i * self.beamwidth), math.sin(i * self.beamwidth)])
            for i in range(self.n_beams)
        ]
        self.current_beam_index = randint(0, self.n_beams - 1)
        self.reference_vector = self.beam_vectors[self.current_beam_index]

    def move_reference_vector(self):
        # Move to the next beam direction in a circular fashion
        self.current_beam_index = (self.current_beam_index + 1) % self.n_beams
        self.reference_vector = self.beam_vectors[self.current_beam_index]

    def relative_angle(self, v1, v2):
        # More efficient and stable than arccos-based method
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)
        return abs(math.atan2(cross, dot))

    def beamforming_gain(self, target_pos, reference_pos):
        # Assumes target_pos and reference_pos are NumPy arrays or list-like
        direction_vector = np.array(target_pos) - np.array(reference_pos)

        angle = self.relative_angle(direction_vector, self.reference_vector)
        if angle <= self.beamwidth / 2:
            return self.main_lobe_gain
        else:
            return self.side_lobe_gain




class Obj:
    def __init__(self, bs, id:int, position:np.ndarray, region:np.ndarray) -> None:
        self.id = id #fr'${id}$'
        self.bs = bs
        self.region = region
        self.position = position
        self.motion = False
        self.velocity = None
        #self.handover_log = [] 
        self.theta = []

        # self.h_pos = []
        # self.v_pos = []
    def init_parameters(self, velox_domain:list, motionless:bool=None): 

        if motionless:
            self.motion = False
        else:
            self.motion = True
            # if coin_toss():
            #     self.motion = True
            # else:
            #     self.motion = False

        if self.motion:
            self.velocity = uniform(low=min(velox_domain), high=max(velox_domain), size=(1,2))[0]
        else:
            self.velocity = [0,0]

        self.position = get_array(self.position)
        self.velocity = get_array(self.velocity)
        self.theta =[merge_vectors(v1=self.position, v2=self.velocity)]
        self.d = len(self.theta[0])
        
    
    def get_distance_from_border(self):
        pp = Polygon(self.region)
        # print(pp, type(pp))
        return pp.exterior.distance(Point(self.position))
    
    # def update_history(self):
        # self.h_pos.append(pos)
        # self.v_pos.append(velox)
        
        #self.theta.append(merge_vectors(v1=self.position, v2=self.velocity))
        
class User(Obj):
    def __init__(self, bs, id:int, position:np.ndarray, region:np.ndarray) -> None:
        # Call the constructor of the parent class Obj
        super().__init__(bs=bs, id=id, position=position, region=region)
        self.comm_sinrs = []
    
    def set_motion_params(self, scaler:float=1):
        self.A = get_A_process(parameter=0.9)
        self.noise_mean = np.zeros(shape=self.d)
        self.noise_cov = get_Z_process(parameter=scaler, dim=self.d)
    
    #@Called by the manager
    def start(self, args):
        self.init_parameters(velox_domain=args["velox_domain"], motionless = args["user_motionless"])
        self.arrival_rate = args["arrival_rate"]
        self.set_motion_params(scaler=args["scaler"]) 
        self.queue = QUEUE() #randint(low=1, high=10)
        self.thr = args["thr_ue"]
        self.n = 1
        #self.plotter = Plotter(kf_manager=None, queue_manager = self, thr=args["thr_ue"])   

    def motion_step(self, t:int):
        #can be according whatever distribution, now GMM
        z_t = multivariate_normal(mean = self.noise_mean ,cov=self.noise_cov , size=self.n).reshape(self.d, self.n)
        step = np.dot(self.A, self.theta[t-1].reshape(self.d, self.n)) + z_t
        step = step.flatten()
        return step
    
    def move(self, t:int):
        #Apply a GMM model
        #if self.motion:
        self.theta.append(self.motion_step(t=t))
        self.position, self.velocity = self.theta[-1][:2], self.theta[-1][2:]
        
    def update_queue(self, t:int):
        
        shannon_rate, sinr = self.bs.shannon_rate(obj=self)
        self.comm_sinrs.append(sinr)
        arrival_rate = self.arrival_rate #now is constant maybe after it gonna vary

        self.queue.update_workload(shannon_rate=shannon_rate,arrival_rate=arrival_rate, t=t)
    
    def finish(self, show_samples:bool):
        self.theta = get_array(self.theta)
        if show_samples:
            self.plotter = Plotter(kf_manager=None, queue_manager = self, thr=self.thr, num_points=100)   
            # self.plotter.activate_buffers()

    
class Sensed_Obj(Obj):
    def __init__(self, bs, id:int, position:np.ndarray, region:np.ndarray) -> None:
        # Call the constructor of the parent class Obj
        super().__init__(bs=bs, id=id, position=position, region=region) 
        self.sensing_sinrs = [] 
    
    def get_interference(self):
        return self.bs.get_interference(obj=None, add_noise=True)

    def get_rx_radar_signal_power(self):
        return self.bs.rx_radar_signal_power(obj_pos=self.position) 

    def compute_noise_covariance_scaler(self)->dict:
        interf_and_noise = self.get_interference()
        signal_pw = self.get_rx_radar_signal_power()
        sinr_rad = cap_sinr(signal_pw / interf_and_noise)
        self.sensing_sinrs.append(sinr_rad)
        return {
            'interference': interf_and_noise,
            'dim': self.meas_dim,
            'signal_power': signal_pw,
        }

    
    #we make two initialization function cause the creation is operated by the VC manager
    def start(self,args):
        self.init_parameters(velox_domain=args["velox_domain"], motionless = args["so_motionless"])

        if self.motion:
            self.orbital_motion = True
            # if coin_toss():
            #     self.orbital_motion = True
            # else:
            #     self.orbital_motion = False

        self.kf_params = {}
        self.observations = []
        self.meas_dim = args["obs_dim"]
        self.init_kf_params(args=args)
        self.thr = args["thr_so"]
        #self.plotter = Plotter(kf_manager=self, queue_manager = None, thr=args["thr_so"])

    def init_kf_params(self, args):
        self.kf_params['C'] = [get_C_process(parameter= args["obs_scaler"], dim=(self.meas_dim, self.d))]
        self.kf_params['A'] = [get_A_process(parameter = args["motion_scaler"])]
        self.kf_params['Z'] = [get_Z_process(parameter = args["motion_noise_scaler"], dim = self.d)]
        self.kf_params['V'] = [get_V_process(args=self.compute_noise_covariance_scaler())]
        self.n = 1

        mu0 = np.array(self.theta).reshape(self.d, self.n) 
        Sigma0 = np.identity(n=self.d, dtype=float)*args["sigma0"]
        mean_z0 = np.zeros(shape=self.d)
        mean_v0 = np.zeros(shape=self.meas_dim)

        self.kf_params['mu'] = [mu0]
        self.kf_params['Sigma']= [Sigma0]
        self.kf_params['mean_z'] = [mean_z0]
        self.kf_params['mean_v'] = [mean_v0] 

        self.kf = KalmanFilter(id = self.id, mu0=mu0, Sigma0=Sigma0) 

    #@Called by the manager
    def update_parameters(self):
        self.position, self.velocity = self.theta[-1][:2], self.theta[-1][2:]
        #V ...
        self.kf_params['V'].append(get_V_process(args=self.compute_noise_covariance_scaler()))
        #Put the logic to update each parameter if needed
        #i.e. A,C,Z,V
        #A: self.kf_params['A'].append(self.get_A_process())
        #C ....
        #Z ....
 
    def motion_step(self):

        A_t = self.kf_params['A'][-1]
        proj_step = np.dot(A_t, self.theta[-1].reshape(self.d, self.n)) 

        if self.orbital_motion:

            bs_center = np.array([self.bs.position[0],self.bs.position[1],1,1])

            A_scl = np.identity(n=self.d) - A_t

            proj_step += np.dot(A_scl, bs_center.reshape(self.d, self.n))

        return proj_step
    
    def move(self):

        #get the latest value of the lists
        mean_z_t = self.kf_params['mean_z'][-1]
        Z_t = self.kf_params['Z'][-1]
        mean_v_t = self.kf_params['mean_v'][-1]
        V_t = self.kf_params['V'][-1]
        C_t = self.kf_params['C'][-1]

        z_t = multivariate_normal(mean = mean_z_t,cov= Z_t, size=self.n).reshape(self.d, self.n)
        v_t = multivariate_normal(mean = mean_v_t, cov= V_t,size=self.n).reshape(self.meas_dim, self.n)

        proj_step = self.motion_step()
        theta_t = (proj_step + z_t).flatten()
        self.theta.append(theta_t)

        y_t = np.dot(C_t, theta_t.reshape(self.d, self.n)) + v_t
        
        #shapes 2*n + 2*n
        self.observations.append(y_t.flatten())
    
    def update_estimate(self):
        self.kf.update_estimate(
            A=self.kf_params['A'][-1], C = self.kf_params['C'][-1],
            V = self.kf_params['V'][-1], Z = self.kf_params['Z'][-1],
            y= self.observations[-1].reshape(self.meas_dim, self.n)
            )
    
    def finish(self, show_samples:bool):
        self.theta = get_array(self.theta)
        self.observations = get_array(self.observations)
        self.kf.estimated_states = get_array(self.kf.estimated_states)
        self.kf.covariance_states = get_array(self.kf.covariance_states)
        if show_samples:
            self.plotter = Plotter(kf_manager=self, queue_manager = None, thr=self.thr, num_points=100)
            # self.plotter.activate_buffers()
        
class BS:
    def __init__(self, id:int, position:np.ndarray, region:np.ndarray=None) -> None:

        self.id = id #fr'${id}$'
        self.position = position
        self.region = region

        self.steady_state_kf_wload = []
        self.region_mark = None

        self.sos_ = []
        self.ues_ = []
        self.neighb_bss = []
        self.mode = None

        self._init_radio_params()
    
    def start(self, neighbors: list, mode:str):
        self.neighb_bss = neighbors
        self.mode = mode
    
    def lin_operator(self, x_db):
        return 10**(x_db / 10)

    def db_operator(self, x_lin):
        return 10 * np.log10(x_lin)

    def rayleigh_fading(self):
        return exponential(scale=1, size=1)[0]
    
    def beamforming_gain(self, target_pos):
        return self.beamformer.beamforming_gain(reference_pos=self.position, target_pos=target_pos)

    def gain(self, obj_pos):
        fading = self.rayleigh_fading()
        gain_db = self.beamforming_gain(target_pos=obj_pos)
        return fading * G_0 * self.lin_operator(gain_db) * torus_distance(x=self.position, y=obj_pos) ** (-ALPHA)

    def radar_gain(self, obj_pos):
        fading = self.rayleigh_fading()
        gain_db = self.beamforming_gain(target_pos=obj_pos)
        return fading * self.lin_operator(gain_db) * self.radar_coeff * torus_distance(x=self.position, y=obj_pos) ** (-4)

    def rx_signal_power(self, obj_pos):
        return self.gain(obj_pos) * self.bs_ptx

    def rx_radar_signal_power(self, obj_pos):
        return self.radar_gain(obj_pos) * self.bs_ptx

    def _init_radio_params(self):
        self.beamformer = DirectionalBeamformer(main_lobe_gain=(G_TX_MAIN+G_RX), side_lobe_gain=G_TX_SIDE, log2_beams=4)
        self.signal_wl = C / F_C
        self.radar_coeff = ((self.signal_wl**2)*exponential(1))/((4*math.pi)**3)
        # Noise and power
        self.noise = self.lin_operator(N)*W #lin(dBm/Hz) * Hz = mW
        self.bs_ptx = self.lin_operator(PW_TX) #lin(dBm) = mW

    def get_interference(self,obj:Obj = None, add_noise:bool=True)-> float:

        if len(self.neighb_bss) == 0:
            raise ValueError("Neighbor base stations list is empty.")

        i_com = 0
        i_sen = 0
        interference = 0

        if self.mode != 'com':
            i_sen = sum(bs.rx_signal_power(obj_pos=self.position) for bs in self.neighb_bss) #+ self.noise
            # i_sen = sum(bs.gain(obj_pos=self.position)*(bs.bs_ptx) for bs in self.neighb_bss)
        
        if self.mode != 'sen':
            if obj is None: #means it is an SO requesting the interference, if self.ues is an empty list this remains 0
                i_com = sum(
                    sum(bs.rx_signal_power(obj_pos=ue.position) for bs in self.neighb_bss)
                    for ue in self.ues_
                    )  
            else:
                i_com = sum(bs.rx_signal_power(obj_pos=obj.position) for bs in self.neighb_bss)
            
        interference = i_sen + i_com
        if add_noise:
            interference += self.noise 
        
        return interference
    
    def get_sinr(self, obj:Obj):
        interf = self.get_interference(obj=obj, add_noise=True)
        # print('UE', obj.id, 'Interf:', interf)
        if interf == 0:
            sinr = float('inf')
        else:
            sinr = self.rx_signal_power(obj_pos=obj.position)/interf
        return cap_sinr(sinr)
    
    def shannon_rate(self, obj:Obj):
        sinr = self.get_sinr(obj = obj)
        # print('sinr', sinr)
        sh = W*np.log2(1+sinr)*10**(-6)
        return sh, sinr #get Mbits/s
        
    # def set_sos(self, sos_list: list) -> None:
    #     self.sos_ = sos_list
    
    # def set_ues(self, ues_list: list) -> None:
    #     self.ues_ = ues_list

    

    # Set the Kalman filter, if multiple sos the bs sets as many kfs instances
    # def set_kf(self, kf) -> None:
    #     """ Set the Kalman Filter (kf) for the base station. """
    #     self.kf = kf
    
    
    

        # scale_factor_fad = np.sqrt(self.lin_operator(SIGMA_FADING_SQRD)/2)
        # self.rayleigh_fading = lambda : ((normal(loc=MU_FADING, scale=scale_factor_fad, size=1)**2) 
        #                                 + (normal(loc=MU_FADING, scale=scale_factor_fad, size=1)**2))[0]
        
        
        # self.lin_rayleigh_fading = lambda : self.lin_operator(-self.rayleigh_fading()) #it considered a loss component

        

        
    # def check_interf(self, obj:Obj):
    #     # Calculate distances from neighboring base stations to the object
    #     distances = {
    #         bs.id: bs.l2_norm(x=bs.position, y=obj.position)
    #         for bs in self.neighb_bss
    #     }
        
    #     # Calculate distance attenuation for each base station
    #     distance_attenuation = {
    #         bs.id: dist**(-ALPHA)
    #         for bs, dist in zip(self.neighb_bss, distances.values())
    #     }
        
    #     # Calculate gains for each base station
    #     gains = {
    #         bs.id: bs.gain(obj_pos=obj.position) 
    #         for bs in self.neighb_bss
    #     }
        
    #     # Calculate receiving powers for each base station
    #     receiving_powers = {
    #         bs.id: gain * bs.bs_ptx
    #         for bs, gain in zip(self.neighb_bss, gains.values())
    #     }
        
        # Output the calculated values
        # if self.verbose:
        #     digits = 1
        #     print('Distances:', dict_formatter(distances,digits), 'm')
        #     print('Distance Attenuation:', dict_formatter(distance_attenuation,digits))
        #     print('Gains:', dict_formatter(gains,digits))
        #     db_power_transformer = {k: formatter(self.db_operator(v),digits) for k, v in receiving_powers.items()}
        #     print('Receiving Powers:', '\n',dict_formatter(receiving_powers,digits), 'mW', '\n',db_power_transformer,'dBm')

    
    
    

