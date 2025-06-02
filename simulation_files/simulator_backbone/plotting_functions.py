from simulation_files.utils.helper_functions import *

def adaptive_step(timeline: float, num_points: int) -> int:
    if num_points == 0:
        raise ValueError("num_points must be greater than zero")
    return floor(len(timeline) / num_points)


def _subsample(indices, data):
        """
        General helper to subsample the timeline and corresponding data using adaptive step.
        If data is provided, returns a tuple (sub_timeline, sub_data), otherwise just sub_timeline.
        """
        sub_data = get_array([data[i] for i in indices])
        return sub_data

def len_printer(lst:list, name:str):
    print(f"{name}: {len(lst)}")

def len_checker(lst_list):
    """
    Given a list of lists, trims each list (by removing the first few elements) so that they
    all have the same length equal to the minimum length.
    """
    # Compute the minimum length among the lists.
    min_length = min(len(lst) for lst in lst_list)
    new_lists = []
    for lst in lst_list:
        if len(lst) != min_length:
            # Remove the excess elements from the beginning.
            delta = len(lst) - min_length
            new_lists.append(lst[delta:])
        else:
            new_lists.append(lst)
    return new_lists

class Plotter:
    def __init__(self, thr: float, num_points:int, kf_manager=None, queue_manager=None) -> None:
        self.thr = thr
        self.kf_manager = kf_manager
        self.queue_manager = queue_manager
        self._initialize_buffers()
        self.indices = list(range(0, len(self.timeline), adaptive_step(timeline=self.timeline, num_points=num_points)))
        self.sub_timeline = [self.timeline[i] for i in self.indices]
    
    def _initialize_buffers(self):
        if self.kf_manager:
            self._init_kf_buffers()
        if self.queue_manager:
            self._init_queue_buffers()

    def _init_kf_buffers(self):

        true_states = self.kf_manager.theta
        observations = self.kf_manager.observations
        estimated_states = self.kf_manager.kf.estimated_states
        covariance_states = self.kf_manager.kf.covariance_states
        stability_states = self.kf_manager.kf.stability_states
        sinrs = self.kf_manager.sensing_sinrs

        # Adjust the lengths so that they all have the same number of elements
        (true_states, observations, estimated_states, covariance_states, stability_states, sinrs) = \
            len_checker([true_states, observations, estimated_states, covariance_states, stability_states, sinrs])

        self.true_states = true_states
        self.observations = observations
        self.estimated_states = estimated_states
        self.covariance_states = covariance_states
        self.stability_states = stability_states
        self.sinrs = sinrs

        self.bs = self.kf_manager.bs
        self.id = self.kf_manager.id
        self.timeline = list(range(len(self.covariance_states)))
        self.cov_norm_states = get_array([matrix_norm(self.covariance_states[t], ord=2) for t in self.timeline])
    
    def _init_queue_buffers(self):
        shannon_rates = self.queue_manager.queue.shannon_rates
        sinrs = self.queue_manager.comm_sinrs  
        wloads = self.queue_manager.queue.wloads
        trace = self.queue_manager.queue.trace

        # Adjust the lengths so that they all have the same number of elements
        (shannon_rates, sinrs, wloads, trace) = \
            len_checker([shannon_rates, sinrs, wloads, trace])
        
        self.shannon_rates = shannon_rates
        self.sinrs = sinrs
        self.wloads = wloads
        self.trace = trace
        
        self.id = self.queue_manager.id
        self.timeline = list(range(len(self.wloads)))
    
    def stability_checker(self, args: dict):
        ax = args['ax']
        eigenvalues = [np.max(np.linalg.eigvals(p)) for p in self.stability_states]
        bound = [1] * len(self.sub_timeline)
        sub_eigenvalues = _subsample(indices=self.indices, data=eigenvalues)
        ax.plot(self.sub_timeline, sub_eigenvalues, color='k', marker='.', linestyle=':', label='Largest Eigenvalue')
        ax.plot(self.sub_timeline, bound, color='r', linestyle=':', label='Stability Bound (1)')
        ax.set(title=r'$\rho(A + L_tC)$', xlabel='Time (s)', ylabel=r'$|\lambda_{max}(A + L_tC)|$')
        ax.legend(title=self.id, loc='best')
        ax.grid(True)
    
    def kf_evolution_1D(self, args: dict):
        ax, coord = args['ax'], args['coord']
        labels = {'x': ('Position', 'X (m)'), 'y': ('Position', 'Y (m)'),
                  'v_x': ('Velocity', 'v_x (m/s)'), 'v_y': ('Velocity', 'v_y (m/s)')}
        if coord.lower() not in labels:
            raise ValueError(f"Invalid coordinate '{coord}'. Must be one of {list(labels.keys())}.")
        title, label = labels[coord.lower()]
        ix = ['x', 'y', 'v_x', 'v_y'].index(coord.lower())
        # steps, steps_ = range(len(self.true_states)), range(len(self.estimated_states))
        # sub_indices = range(0, len(steps), adaptive_step(timeline=steps, num_points=100))
        # sub_indices_ = range(0, len(steps_), adaptive_step(timeline=steps_, num_points=100))
        # sub_steps = [steps[i] for i in sub_indices]
        # sub_steps_ = [steps_[i] for i in sub_indices_]
        # sub_true_states = [self.true_states[i, ix] for i in sub_steps]
        # sub_estimated_states = [self.estimated_states[i, ix] for i in sub_steps_]

        sub_true_states = _subsample(indices=self.indices, data=self.true_states[:,ix])
        sub_estimated_states = _subsample(indices=self.indices, data=self.estimated_states[:,ix])
        
        if title == 'Position':
            ax.scatter(0, self.bs.position[ix], marker='^', color='b', label='Base station')
        ax.plot(self.sub_timeline, sub_true_states, label=r'$\theta_t$', linestyle='--', marker=".", linewidth=1)
        ax.plot(self.sub_timeline, sub_estimated_states, label=r'$\hat{\theta}_t$', linestyle=':', marker=".", linewidth=1)
        ax.set(xlabel='Time (s)', ylabel=label, title=f'True state vs KF estimate ({title})')
        ax.legend(title=self.id.upper())
        ax.grid(True)
    
    def covariance_evolution(self, args: dict):
        ax = args['ax']
        bound = [self.thr] * len(self.sub_timeline) 
        sub_cov_norm_states = _subsample(indices=self.indices, data=self.cov_norm_states)
        ax.plot(self.sub_timeline, sub_cov_norm_states, label=self.id, color='k', linestyle=':', marker='.', linewidth=1.5)
        ax.plot(self.sub_timeline, bound, color='r', linestyle=':', label=f'Threshold ({self.thr})')
        ax.set(xlabel='Time (s)', ylabel=r'$\sigma_{max}(\Sigma_{t})$', title=r'KF Error Covariance $\ell_2$-norm Evolution')
        ax.legend(title=self.id.upper())
        ax.grid(True)
    
    def kf_error_process(self, args: dict):
        ax = args['ax']
        sub_estimated_states = _subsample(indices=self.indices, data=self.estimated_states)
        sub_true_states = _subsample(indices=self.indices, data=self.true_states)
        error = [l2_norm(ts, es) for es, ts in zip(sub_estimated_states, sub_true_states)]
        ax.plot(self.sub_timeline, error, label=r'$||\hat \theta_t - \theta_t||_2$', color='r', linestyle=':', marker='.', linewidth=1.5)
        ax.set(title=r'KF Absolute Error Evolution', xlabel='Time (s)', ylabel='KF Error')
        ax.legend(title=self.id.upper())
        ax.grid(True)
    
    def queue_evolution(self, args: dict):
        ax = args['ax']
        sub_wloads = _subsample(indices=self.indices, data=self.wloads)
        ax.plot(self.sub_timeline, sub_wloads, label=f'Workload', color='k', marker='.', linestyle=':')
        if self.thr is not None:
            ax.axhline(y=self.thr, color='red', linestyle=':', label=f'Threshold ({self.thr})')
        ax.set(title='Queue Workload Evolution', xlabel='Time (s)', ylabel='Workload')
        ax.legend(title=self.id.upper())
        ax.grid(True)
    
    def sinr_evolution(self, args: dict):
        ax = args['ax']
        tag = 'com' if self.kf_manager is None else 'rad'
        sub_sinrs = _subsample(indices=self.indices, data=self.sinrs)
        ax.plot(self.sub_timeline, sub_sinrs, label=rf'$SINR_{{{tag}}}$', color='C2', marker='.', linestyle=':')
        ax.set(title=rf'$SINR_{{{tag}}}$ Evolution', xlabel='Time (s)', ylabel=rf'$SINR_{{{tag}}}$')
        ax.legend(title=self.id.upper())
        ax.grid(True)
    
    def lindley_evolution(self, args: dict):
        ax = args['ax']
        # timeline = sorted(self.trace.keys())
        arrivals, serviced = zip(*[self.trace[t] for t in self.timeline])
        sub_arrivals = _subsample(indices=self.indices, data=arrivals)
        sub_serviced = _subsample(indices=self.indices, data=serviced)
        ax.plot(self.sub_timeline, sub_arrivals, label=f'Arrivals', color='C1', marker='.')
        ax.plot(self.sub_timeline, sub_serviced, label=f'Serviced', color='C0', marker='.')
        if self.thr is not None:
            ax.axhline(y=self.thr, color='red', linestyle='--', 
                    label=f'Threshold (thr={self.thr})')
        ax.set(title='Packets Evolution', xlabel='Time (s)', ylabel='Load')
        ax.legend(title=self.id.upper())
        ax.grid(True)

    #OLD 
    def shannon_rate_evolution(self, args: dict):
        ax = args['ax']
        sub_shannon_rates = _subsample(indices=self.indices, data=self.shannon_rates)
        ax.plot(self.sub_timeline, sub_shannon_rates, label=f'Shannon Rate', color='C2', marker='.', linestyle=':')
        ax.set(title='Shannon Rate Evolution', xlabel='Time (s)', ylabel='Mbit/s')
        ax.legend(title=self.id.upper())
        ax.grid(True)

    def covariance_distribution(self, args: dict):
        ax = args['ax']
        sns.kdeplot(self.cov_norm_states, fill=True, color='orange', ax=ax, label=self.id)
        ax.axvline(x=self.thr, color='red', linestyle='--', label=f'Threshold ({self.thr})')
        ax.set(title=r'KF Covariance $\ell_2$-norm Distribution', xlabel='Values', ylabel='Density')
        ax.legend(title=self.id.upper())


# class Plotter():
#     def __init__(self, kf_manager = None, queue_manager = None, thr:float = None) -> None:
#         self.thr = thr
#         self.kf_manager = kf_manager
#         self.queue_manager = queue_manager

#     def activate_buffers(self):
#         if self.kf_manager is not None:
#             self._init_kf_buffers()
#         if self.queue_manager is not None:
#             self._init_queue_buffers()
       
#     def _init_kf_buffers(self):
#         # print(self.kf_manager.theta)
#         self.true_states = self.kf_manager.theta
#         self.observations = self.kf_manager.observations
#         self.estimated_states = self.kf_manager.kf.estimated_states
#         self.covariance_states = self.kf_manager.kf.covariance_states
#         self.stability_states = self.kf_manager.kf.stability_states
#         self.bs = self.kf_manager.bs
#         self.id = self.kf_manager.id
#         self.timeline = range(len(self.covariance_states))
#         self.cov_norm_states = get_array([matrix_norm(self.covariance_states[t], ord=2) for t in self.timeline])
#         # print(self.cov_norm_states)
    
#     def _init_queue_buffers(self):
#         self.shannon_rates = self.queue_manager.queue.shannon_rates  
#         self.id = self.queue_manager.id
#         self.wloads = self.queue_manager.queue.wloads  # Workload data
#         self.trace = self.queue_manager.queue.trace

    
#     def stability_checker(self,ax):
#         # Compute the eigenvalues and eigenvectors
#         eigenvalues = []
#         for p in self.stability_states:
#             eigenval, _ = np.linalg.eig(p)
#             eigenvalues.append(np.max(eigenval))

#         # Plot the largest eigenvalue over time
#         timeline = list(range(len(eigenvalues)))  # Convert range to list
#         bound = [1] * len(timeline)  # Properly create a list of 1's
#         ll = self.id #f'i={self.id}'
#         # ax.plot(timeline, eigenvalues, color='k', marker='.', linestyle='--', label='Largest Eigenvalue')
#         ax.scatter(timeline, eigenvalues, color='k', marker='.', label='Largest Eigenvalue')
#         ax.plot(timeline, bound, color='r', linestyle=':',label='Stability Bound (1)')
#         ax.set_title(r'$\rho(A + L_tC)$')
#         ax.set_xlabel(r'$t$ (s)')
#         ax.set_ylabel(r'$|\lambda_{max}(A + L_tC)|$')
#         ax.legend(title=ll, loc='best')
#         # Update legend to include explanations
#         # Adding detailed legend explanations
#         ax.grid(True)
#         # ax.show()

#     def kf_evolution_1D(self, ax, coord: str):
#         """
#         Plot the 1D evolution of a Kalman Filter.

#         Parameters:
#             ax (matplotlib.axes.Axes): The axis to plot on.
#             coord (str): The coordinate to plot ('x', 'y', 'v_x', 'v_y').
#         """
#         # Define labels for true state and estimate
#         label_t = r'$\theta_t$'
#         label_e = r'$\hat{\theta}_t$'

#         # Determine the index and labels based on the coord argument
#         match coord.lower():
#             case 'x':
#                 ix = 0
#                 title = 'Position'
#                 label = 'X (m)'
#             case 'y':
#                 ix = 1
#                 title = 'Position'
#                 label = 'Y (m)'
#             case 'v_x':
#                 ix = 2
#                 title = 'Velocity'
#                 label = 'v_x (m/s)'
#             case 'v_y':
#                 ix = 3
#                 title = 'Velocity'
#                 label = 'v_y (m/s)'
#             case _:
#                 raise ValueError(f"Invalid coordinate '{coord}'. Must be one of 'x', 'y', 'v_x', 'v_y'.")

#         # Define the steps for plotting
#         steps = range(len(self.true_states))
#         steps_ = range(len(self.estimated_states))

#         if title == 'Position':
#             ax.scatter(0, self.bs.position[ix], marker='^', color='b', label='Base station')

#         # Plot true states and Kalman Filter estimates
#         ax.plot(steps, self.true_states[:, ix], label=f'True state ({label_t})', linestyle='--', marker=".", linewidth=1)
#         ax.plot(steps_, self.estimated_states[:, ix], label=f'KF estimate ({label_e})', linestyle=':', marker=".", linewidth=1)

#         # Set axis labels, title, legend, and grid
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel(label)
#         ax.set_title(f'True state ({label_t}) vs KF estimate ({label_e}) ({title})')
#         ax.legend(title=self.id.upper())
#         ax.grid(True)


#     # def kf_evolution_1D(self, ax, coord:str):
#     #     # Plot true position, noisy observations, and estimated position
#     #     label_t = r'$\theta_t$'
#     #     label_e = r'$\hat{\theta}_t$'
#     #     #label_meas = r'$y_t$'
#     #     #n_steps = len(self.true_states)
        
#     #     # cmap = plt.get_cmap('viridis', n_steps)
#     #     # s = n_steps/10
#     #     # for i,t in enumerate(range(n_steps)):
#     #     #     color = cmap(i / (n_steps - 1))
#     #     #     ax.plot(self.true_states[i, 0], self.true_states[i, 1], color=color, label=f't = {t}' if i % s == 0 else "", linestyle='--', marker=".", linewidth=1)

#     #     ax.scatter(0, self.bs.position[0], marker='^', color='b')

#     #     steps = range(len(self.true_states))
#     #     steps_ = range(len(self.estimated_states))

#     #     case coord.lower() == 'x':
#     #         ix = 0
#     #         title = 'position'
#     #         label = 'X (m)'
#     #     case coord.lower() == 'y':
#     #         ix = 1
#     #         title = 'position'
#     #         label = 'Y (m)'
#     #     case coord.lower() == 'v_x':
#     #         ix = 2
#     #         title = 'velocity'
#     #         label = 'v_x (m/s)'
#     #     case coord.lower() == 'v_y':
#     #         ix = 3
#     #         title = 'velocity'
#     #         label = 'v_y (m/s)'

#     #     ax.plot(steps, self.true_states[:, ix], label=f'True state ({label_t})', linestyle='--', marker=".", linewidth=1)
#     #     ax.plot(steps_, self.estimated_states[:, ix], label=f'KF estimate ({label_e})', linestyle=':', marker=".", linewidth=1)
#     #     #ax.plot(self.observations[:, 0], self.observations[:, 1], label=f'Noisy measurement ({label_meas})', marker='o', linestyle='none', alpha=0.5)

#     #     ax.set_xlabel('time (s)')
#     #     ax.set_ylabel(label)
#     #     ax.set_title(f'True state ({label_t}) vs KF estimate ({label_e}) ({title})')
#     #     ax.legend(title=self.id.upper())
#     #     ax.grid(True)

#         # ax.show()
#     # def kf_velocity_evolution_1D(self,ax):
#     #     # Plot true position, noisy observations, and estimated position
#     #     label_t = r'$\theta_t$'
#     #     label_e = r'$\hat{\theta}_t$'

#     #     steps = range(len(self.true_states))
#     #     steps_ = range(len(self.estimated_states))

#     #     ax.plot(steps, self.true_states[:, 2],  label=f'True state ({label_t})', linestyle='--', marker=".", linewidth=1)
#     #     #ax.plot(self.observations[:, 0], self.observations[:, 1], label='Noisy measurement', marker='o', linestyle='none', alpha=0.5)
#     #     ax.plot(steps_, self.estimated_states[:, 2], label=f'KF estimate ({label_e})', linestyle=':', marker=".", linewidth=1)

#     #     #print('estimated states:',self.estimated_states[:, 2], self.estimated_states[:, 3] )
#     #     ax.set_xlabel('time (s)')
#     #     ax.set_ylabel('v_x (m/s)')
#     #     ax.set_title(f'True state ({label_t}) vs KF estimate ({label_e}) (velocity)')
#     #     ax.legend(title=self.id.upper())
#     #     ax.grid(True)


#     # def kf_position_evolution_2D(self, ax):
#     #     # Plot true position, noisy observations, and estimated position
#     #     label_t = r'$\theta_t$'
#     #     label_e = r'$\hat{\theta}_t$'
#     #     #label_meas = r'$y_t$'
#     #     #n_steps = len(self.true_states)
        
#     #     # cmap = plt.get_cmap('viridis', n_steps)
#     #     # s = n_steps/10
#     #     # for i,t in enumerate(range(n_steps)):
#     #     #     color = cmap(i / (n_steps - 1))
#     #     #     ax.plot(self.true_states[i, 0], self.true_states[i, 1], color=color, label=f't = {t}' if i % s == 0 else "", linestyle='--', marker=".", linewidth=1)

#     #     ax.scatter(self.bs.position[0],self.bs.position[1], marker='^', color='b')

#     #     ax.plot(self.true_states[:, 0], self.true_states[:, 1], label=f'True state ({label_t})', linestyle='--', marker=".", linewidth=1)
#     #     #ax.plot(self.observations[:, 0], self.observations[:, 1], label=f'Noisy measurement ({label_meas})', marker='o', linestyle='none', alpha=0.5)
#     #     ax.plot(self.estimated_states[:, 0], self.estimated_states[:, 1], label=f'KF estimate ({label_e})', linestyle=':', marker="d", linewidth=1)

#     #     ax.set_xlabel('X (m)')
#     #     ax.set_ylabel('Y (m)')
#     #     ax.set_title(f'True state ({label_t}) vs KF estimate ({label_e}) (position)')
#     #     ax.legend(title=self.id.upper())
#     #     ax.grid(True)

#     #     # ax.show()
#     # def kf_velocity_evolution_2D(self,ax):
#     #     # Plot true position, noisy observations, and estimated position
#     #     label_t = r'$\theta_t$'
#     #     label_e = r'$\hat{\theta}_t$'

#     #     ax.plot(self.true_states[:, 2], self.true_states[:, 3],  label=f'True state ({label_t})', linestyle='--', marker=".", linewidth=1)
#     #     #ax.plot(self.observations[:, 0], self.observations[:, 1], label='Noisy measurement', marker='o', linestyle='none', alpha=0.5)
#     #     ax.plot(self.estimated_states[:, 2], self.estimated_states[:, 3], label=f'KF estimate ({label_e})', linestyle=':', marker="d", linewidth=1)

#     #     #print('estimated states:',self.estimated_states[:, 2], self.estimated_states[:, 3] )
#     #     ax.set_xlabel('v_x (m/s)')
#     #     ax.set_ylabel('v_y (m/s)')
#     #     ax.set_title(f'True state ({label_t}) vs KF estimate ({label_e}) (velocity)')
#     #     ax.legend(title=self.id.upper())
#     #     ax.grid(True)


#     def covariance_evolution(self, ax):

#         #self.cov_norm_states = get_array([matrix_norm(self.covariance_states[t], ord=2) for t in self.timeline])
#         bound = [self.thr] * len(self.timeline)  # Properly create a list of thr's

#         # label = r'$||\Sigma_{i,t}||_2$'
#         # tt = r'$\forall t$'
#         # ll = f'i={self.id}'
#         label = r'KF Covariance $l_2$-norm Temporal Evolution'
#         # tt = r'$\forall t$'
#         ll = self.id
#         ax.plot(self.timeline, self.cov_norm_states, label=ll,color='k', linestyle='--', marker='.', linewidth=1.5)
#         ax.plot(self.timeline, bound, color='r', linestyle=':',label=f'Threshold ({self.thr})')
        
#         ax.set_xlabel(r'$t$ (s)')
#         # ax.set_ylabel(r'$\sigma_{max}(\Sigma_{i,t})$')
#         ax.set_ylabel(r'$\sigma_{max}(\Sigma_{t})$')
#         ax.set_title(label) #f'{label} {tt}'
#         ax.legend()
#         ax.grid(True)
#         # ax.show()
    
#     def covariance_distribution(self, ax):
#         # print(cov_norm_states)

#         # label = r'$f(\{||\Sigma_{i,t}||_2\}_{t=0}^{T-1})$'
#         # l2 = r'$\{||\Sigma_{i,t}||_2\}_{t=0}^{T-1}$'
#         #l2 = r'$||\Sigma_{i}||_2$' 

#         #tt = r'$\forall t$'
#         #self.cov_norm_states = get_array([matrix_norm(self.covariance_states[t], ord=2) for t in self.timeline])
#         ll = self.id
#         label = r'KF Covariance $l_2$-norm Distribution'
#         l2 = 'Values'
#         sns.kdeplot(self.cov_norm_states, fill=True, color='orange',ax=ax , label=ll)
#         ax.axvline(x=self.thr, color='red', linestyle='--', label=f'Threshold ({self.thr})')
#         ax.set_title(f'{label}')
#         ax.set_xlabel(l2)
#         ax.set_ylabel('Density')
#         ax.legend()

#     def kf_prb_thr(self, ax):
#         """Computes the average number of times the norm of the Cov matrix surpassed the given threshold."""

        
#         #self.cov_norm_states = get_array([matrix_norm(self.covariance_states[t], ord=2) for t in self.timeline])
#         above_threshold = (self.cov_norm_states >= self.thr).astype(int)

#         # Compute the probability of exceeding the threshold for each time step

#         prob_above_threshold = above_threshold.mean() #discrete rv
        
#         label = r'$P(||\Sigma_{i,t}||_2$' +  rf'$\geq {self.thr})$'
#         tt = r'$\forall t$'
#         # ll = f'i={self.id}'
#         ll = self.id
#         ax.stem(self.timeline, above_threshold, label=ll, linefmt = 'k:')# color='r')
#         #ax.scatter(self.timeline, above_threshold, label=ll, marker = '.',color='k')
#         ax.set_xlabel(r'$t$ (s)')
#         ax.set_ylabel(f'Probability')
#         ax.set_title(f'{label} {tt}')
#         ax.set_ylim(bottom=-0.1, top=1.1)
#         ax.grid(True)
#         ax.legend()

#         return prob_above_threshold
    
#     def kf_error_process(self, ax):
        
#         swr_error = [l2_norm(ts,es) for es,ts in zip(self.estimated_states, self.true_states)]
#         label = r'$||\hat \theta(t) - \theta(t)||_2$'

#         ax.plot(self.timeline, swr_error, label=label, color='r', linestyle='--', marker='.', linewidth=1.5)

#         ax.set_title('KF Error') 
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('KF Error')
#         ax.legend()
#         ax.grid(True)

    
#     def queue_evolution(self, ax):
#         """
#         Plots the evolution of load as a function of time.

#         Args:
#             ax (matplotlib.axes.Axes): The Axes object to draw the plot on.
#         """
#         timeline = range(len(self.wloads)) 
#         # Plot workload evolution
#         ax.plot(timeline, self.wloads, label=f'{self.id} - Workload', color='k', marker='.', linestyle=':')
#         #ax.scatter(timeline, self.wloads, label=f'{self.id} - Workload', color='C1', marker='.')#linestyle=':'

#         # Add threshold line if provided
#         if self.thr is not None:
#             ax.axhline(y=self.thr, color='red', linestyle='--', label=f'Threshold (thr={self.thr})')

#         # Customize plot
#         ax.set_title('Queue Workload Evolution') #{self.id} -
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Workload')
#         ax.legend()
#         ax.grid(True)
    
#     def shannon_rate_evolution(self, ax):
#         """
#         Plots the evolution of the shannon rate as a function of time.

#         Args:
#             ax (matplotlib.axes.Axes): The Axes object to draw the plot on.
#         """
#         timeline = range(len(self.shannon_rates)) 
#         sub_indices = range(0, len(timeline), 50)
#         new_timeline = [timeline[i] for i in sub_indices]
#         new_shannon_rates = [self.shannon_rates[i] for i in sub_indices]
        
#         # Plot workload evolution
#         # ax.plot(timeline, self.shannon_rates, label=f'{self.id} - Shannon Rate', color='C2', marker='.', linestyle=':')
#         ax.plot(new_timeline, new_shannon_rates, label=f'{self.id} - Shannon Rate', color='C2', marker='.', linestyle=':')

#         # Customize plot
#         ax.set_title('Shannon Rates Evolution') #f'{self.id} - 
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Shannon Rate (Mbit/s)')
#         ax.legend()
#         ax.grid(True)

#     def lindley_evolution(self, ax):
#         """
#         Plots the evolution of the queue's workload (Lindley's evolution) over time.

#         Args:
#             ax (matplotlib.axes.Axes): The Axes object to draw the plot on.
#         """
#         # Sort the time points
#         timeline = sorted(self.trace.keys())

#         # Extract arrivals and serviced values in the order of the sorted timeline
#         arrivals, serviced = zip(*[self.trace[t] for t in timeline])

#         # Select every 50th time point for scatter plots
#         sub_indices = range(0, len(timeline), 50)
#         self.sub_timeline = [timeline[i] for i in sub_indices]
#         sub_arrivals = [arrivals[i] for i in sub_indices]
#         sub_serviced = [serviced[i] for i in sub_indices]

#         # Plot workload evolution
#         ax.plot(self.sub_timeline, sub_arrivals, label=f'{self.id} - Arrivals', color='C1', marker='.',linestyle=':')
#         ax.plot(self.sub_timeline, sub_serviced, label=f'{self.id} - Serviced', color='C2', marker='x',linestyle=':')

        
#         # Compute the difference (arrivals - serviced)
#         #delta_arrivals_serviced = [a - s for a, s in zip(arrivals, serviced)]

#         # Plot workload evolution
#         # ax.scatter(timeline, arrivals, label=f'{self.id} - Arrivals', color='C2', marker='o')
#         # ax.scatter(timeline, serviced, label=f'{self.id} - Serviced', color='C1', marker='x')
#         # ax.plot(timeline, delta_arrivals_serviced, 
#         #         label=f'{self.id} - Delta (Arrivals, Serviced)', 
#         #         color='m', marker='.', linestyle='--')
#         # ax.plot(timeline, self.wloads, 
#         #         label=f'{self.id} - Workload', 
#         #         color='k', marker='.', linestyle='--')

#         # Optionally plot a threshold line if self.thr is set
#         if self.thr is not None:
#             ax.axhline(y=self.thr, color='red', linestyle='--', 
#                     label=f'Threshold (thr={self.thr})')

#         # Customize plot
#         ax.set_title('Queue Workload Evolution')
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('Workload')
#         ax.legend()
#         ax.grid(True)

#     def local_correlation(self, ax, wloads: list = None, cov_norm_states: list = None):
#         # Number of decimal digits for formatting
#         digits = 2

#         # Ensure self.wloads and self.cov_norm_states have the same length
#         wloads = wloads if wloads is not None else self.wloads
#         cov_norm_states = cov_norm_states if cov_norm_states is not None else self.cov_norm_states
#         num_points = min(len(wloads), len(cov_norm_states))
#         wloads = wloads[:num_points]
#         cov_norm_states = cov_norm_states[:num_points]
        
#         if num_points == 0:
#             raise ValueError("wloads and cov_norm_states must contain at least one element each.")

#         # Compute Pearson correlation coefficient

#         #eventually:
#         # min_max_normalizer(avg_covariances)
#         # min_max_normalizer(avg_queue_lengths)

#         correlation = np.corrcoef(wloads, cov_norm_states)[0, 1]

#         # Generate a list of distinct colors using a colormap
#         cmap = cm.get_cmap('viridis')  # 'tab20' has 20 distinct colors
#         #times = np.linspace(0, 1, num_points)
#         times = list(range(num_points))

#         #colors = [cmap(i % 20) for i in range(num_points)]
        
#         # Assume 'time' corresponds to each point; replace with actual time values if available
#         #times = list(range(num_points))
        
#         # Plot scatter with colors representing time
#         scatter = ax.scatter(wloads, cov_norm_states, c=times, cmap=cmap, edgecolors='k', s=30)
        
#         # Add colorbar to represent time
#         # cbar = ax.figure.colorbar(scatter, ax=ax, label='Time (s)')
#         # cbar.set_ticks(range(0, num_points, max(1, num_points // 10)))  # Adjust ticks based on number of points

#         cbar = ax.figure.colorbar(scatter, ax=ax, label='Time (s)')
#         #cbar.set_ticks(np.linspace(0, 1, min(10, num_points)))
#         cbar.set_ticks(np.linspace(min(times), max(times), min(10, num_points)))  # Adjust ticks based on real time range

#         plot_title = f"Pearson Correlation Coefficient: {formatter(correlation,digits)}"
#         ax.set_title(plot_title)

#         # Set axis labels
#         ax.set_xlabel('Queue Workload')
#         ax.set_ylabel(r'Covariance $l_2$-Norm')

#         # Enable grid
#         ax.grid(True)

#         # Add a legend
#         ax.legend(handles=[scatter], title=self.id, loc='upper right')
    