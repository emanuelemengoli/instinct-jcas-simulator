from simulation_files.simulator_backbone.logger import *
from simulation_files.simulator_backbone.controller import *

class Simulation:

    def __init__(self, save_dir:str, bss:list, sos:list, ues:list, t_max:int, verbose:bool, show_samples:bool, mode=str) -> None:
        self.mode = mode.lower()
        self.save_dir = f"{save_dir}/{self.mode}"
        self.bss = bss
        self.t_max = t_max
        self.timeline = np.array(range(self.t_max))

        self._init_manager(sos=sos, ues=ues, verbose=verbose, show_samples=show_samples)
        
        
    def _init_manager(self, sos:list, ues:list, verbose:bool, show_samples:bool):
        #for safety reason is better to force ues or sos to be none
        valid_modes = {'jcas': {'ues': ues, 'sos': sos}, 'sen': {'ues': [], 'sos': sos}, 'com': {'ues': ues, 'sos': []}}
        if self.mode not in valid_modes.keys():
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from {valid_modes}.")
        self.sos = valid_modes[self.mode]['sos']
        self.ues = valid_modes[self.mode]['ues']
        self.netManager = NetManager(bss = self.bss, ues = self.ues, sos = self.sos, n_steps = self.t_max,
                                      verbose = verbose, show_samples = show_samples, save_dir = self.save_dir, mode=self.mode)

    def run(self, args_so:dict, args_ue:dict, args_bs:dict, tail:bool):

        # vc_diam_dict = {bs: get_vc_diam(bs.region) for bs in self.bss}
            
        # Sort by the vc_diam value (the second item in the (bs.id, vc_diam) pair)
        # sorted_vc = sorted(vc_diam_dict.items(), key=lambda x: x[1])  # ascending order by vc_diam
        ## bs_params = dict(sorted(bs_params.items()))
        steady_state_thr = round(0.3*self.t_max)
        thr_ue = args_ue["thr_ue"]
        thr_so = args_so["thr_so"]

        self.netManager.start(args_so=args_so, args_ue=args_ue)

        self.netManager.run(steady_state_thr = steady_state_thr)

        self.generate_env_description(args_so = args_so, args_ue = args_ue, args_bs = args_bs)

        self.plot_output(thr_ue = thr_ue,thr_so = thr_so,steady_state_thr=steady_state_thr)

        if self.mode == 'jcas':
            self.measure_association(thr_ue = thr_ue,thr_so = thr_so,steady_state_thr = steady_state_thr,tail=tail)

            bs_dict = {}

            for bs in self.bss:
                steady_wload_kf = get_array(bs.steady_state_kf_wload)
                wloads, kf_covs = steady_wload_kf[:, 0], steady_wload_kf[:, 1]
                bs_dict[bs] = (np.mean(wloads), np.mean(kf_covs))


            self.scatter_kf_workload(bs_dict = bs_dict, thr_ue=thr_ue,thr_so=thr_so)


    
    def generate_env_description(self, args_so, args_ue, args_bs):

        digits = 2
        lambda_arr = formatter(args_ue['arrival_rate'],digits)
        rho = formatter(args_so["motion_scaler"],digits)
        velox_domain = args_so["velox_domain"]
        thr_ue = args_ue["thr_ue"]
        thr_so = args_so["thr_so"]
        lambda_bs = args_bs['lambda_bss']
        obs_scale = args_so['obs_scaler']
        motion_noise_scaler = args_so['motion_noise_scaler']
        sigma0 = args_so['sigma0']
       
        interf_field = get_array([bs.get_interference(obj=None, add_noise = False) for bs in self.bss]).mean()
        db_operator = self.bss[0].db_operator

        radio_params = {
            r'$N_{BSs}$,$N_{UEs}$,$N_{SOs}$': (len(self.bss),len(self.ues),len(self.sos)),
            'Area': f'{formatter(x=(Wx*Hy*4)/10**6, dig=3)}'+r' $\text{km}^2$',
            r'Central Frequency ($f_c$)' : f'{formatter(F_C/10**9,digits)} GHz',
            r'Noise ($N$)' : f'{formatter(N+db_operator(W),digits)} dBm',
            r'Bandwidth ($W$)' : f'{formatter(W/10**6,digits)} MHz',
            r'Transmitting Power ($P_{tx}$)': f'{PW_TX} dBm',
            r'Free-space path loss ($L_{0}$)': f'{formatter(-db_operator(G_0), digits)}'+r' dB/$d_{0}$ where $d_{0} = 1$ m',
            r'Antenna Gain TX - Main Lobe ($G_{tx}$)': f'{G_TX_MAIN}'+r' dBi',
            r'Antenna Gain TX - Side Lobe ($G_{tx}$)': f'{G_TX_SIDE}'+r' dBi',
            r'Antenna Gain RX (UE) ($G_{tx}$)': f'0 dBi',
            # r'Fading Power Expected value ($\gamma_{fading}$)': r'$ \gamma_{fading} = 10^{P_{tx}(\text{dBm}) / 10}$',
            # r'Fading ($H(t)$)': r'$H(t) \sim Exp(\frac{2}{\gamma_{fading}})$' + r' (Exp. = $\text{Rayleigh}^2$)',
            r'Small Scale Fading ($H(t)$)': r'$H(t) \sim Exp(1)$' + r' (Exp. = $\text{Rayleigh}^2$)',
            r'Path Loss expoent ($\alpha$)': ALPHA,
            r'$\bar \xi(0)$ - Typical BS Interference Filed' : f'{formatter(db_operator(interf_field), digits)} dBm',
            # r'Log-distance Path Loss Model ($P_{L}(d(t))$)': r'$P_{L}(d(t)) = L_{0} + 10\alpha \log _{10}{(\frac {d(t)}{d_{0}})}$',
            # r'Receiving Power at UE ($P_{rx}(t)$)': r'$P_{rx}(t) = P_{tx} \cdot H(t) \cdot \left(P_{L}(d(t))[\text{lin}]\right)^{-1}]$',
            # r"Receiving Power at SO ($P'_{rx}(t)$)": r"$P'_{rx}(t) = P_{tx} \cdot H(t) \cdot \left(P_{L}(d(t))[\text{lin}]\right)^{-1}]$",
        }

        vv = array_formatter(apply_lambda(func=convert_to_km_per_h , lst=velox_domain), dig=2)

        queue_params = {
            r'$\lambda_{BS}$' : f"{lambda_bs}"+r'$\frac{\text{BS}}{\text{km}^2}$',
            r'$\delta$': obs_scale,
            r'$\kappa$': motion_noise_scaler, 
            r'$\sigma_0$': sigma0,
            'Object Velocity':r'$ \mathcal{U} $'+f' ([{vv[0]}, {vv[1]}]) (km/h)',
            'Simulation time': f'{self.t_max} (s)',
            r'$\tau_W$,$\tau_\Sigma$' : (thr_ue,thr_so),
            r'$\forall s \in \mathcal{M}$' : r'$\lambda_{arr}$' + f'= {lambda_arr}, '+r'$\rho$'+ f'= {rho}'
            #Put also the scaler for the UEs
        }

        generate_log(hyperparams=radio_params, title='Radio Environment Parameters', header=['Radio Environment Parameters',"Value"], save_dir=self.save_dir)
        print('\n')

        generate_log(hyperparams=queue_params, title='Simulation Parameters', header=['KF and Queues Parameters','Value'], save_dir=self.save_dir)
        print('\n')

    
    def measure_association(self, thr_ue: float, thr_so: float, steady_state_thr: int, tail: bool = False) -> None:
        """Measures association between workload and state covariance thresholds."""
        digits = 2

        steady_state_timeline = self.timeline[steady_state_thr:]

        avg_covariances = get_array([
                get_temporal_avg([matrix_norm(so.kf.covariance_states[t], ord=2) for t in steady_state_timeline]) 
                for so in self.sos
        ])
        
        avg_wloads = get_array([
                get_temporal_avg([ue.queue.wloads[t] for t in steady_state_timeline]) 
                for ue in self.ues
            ])

        pr_w_thr = get_probability_thr(sample=avg_wloads, thr=thr_ue, tail=tail)

        pr_kf_thr = get_probability_thr(sample=avg_covariances, thr=thr_so, tail=tail)

        joint_prb = get_joint_probability_thr(w_sample=avg_wloads,kf_sample=avg_covariances,
                                              w_thr=thr_ue,kf_thr=thr_so,tail=tail)
        prod_pr = pr_w_thr*pr_kf_thr

        for bs in self.bss:
            steady_wload_kf = get_array(bs.steady_state_kf_wload)
            wloads, kf_covs = get_array(np.mean(steady_wload_kf[:, 0])), get_array(np.mean(steady_wload_kf[:, 1]))
            bs.region_mark = (
                get_probability_thr(sample=wloads, thr=thr_ue, tail=tail),
                get_probability_thr(sample=kf_covs, thr=thr_so, tail=tail)
            )
            # bs.region_mark = (
            #     np.mean(indicator_thr(sample=wloads, thr=thr_ue, tail=tail)),
            #     np.mean(indicator_thr(sample=kf_covs, thr=thr_so, tail=tail))
            # )
        
        str_joint = None
        str_prod = None
        rho_str = None
        # eta_str = None
        thr_str = r"Thresholds ($\tau_W, \tau_\Sigma$)"
        if tail:
            str_joint = r'$\bar F_{U,K}(\tau_\Sigma, \tau_W)$'
            str_prod = r'$\bar F_{U}(\tau_\Sigma) \times \bar F_{K}(\tau_W)$'
            rho_str = r"$\bar \rho_{K,U} (\tau_\Sigma, \tau_W)$"
            # eta_str = r"$\bar \eta(\tau_W, \tau_\Sigma)$"
        else:
            str_joint = r'$F_{U,K}(\tau_\Sigma, \tau_W)$'
            str_prod = r'$F_{U}(\tau_\Sigma) \times F_{K}(\tau_W)$'
            rho_str = r"$\rho_{K,U} (\tau_\Sigma, \tau_W)$"
            # eta_str = r"$\eta(\tau_W, \tau_\Sigma)$"

        
        ratio = joint_prb/prod_pr if prod_pr is not 0 else None
        
        prb_data = {
            thr_str:f'({thr_ue},{thr_so})',
            str_joint: formatter(joint_prb, digits),
            str_prod: formatter(prod_pr, digits),
            rho_str: formatter(ratio, digits)
        }
        # Explanation of the association measure
        print(
            f"[INFO] Association ratio (ρ) = Joint Probability / Product of Marginals\n"
            f"Interpretation: A ratio > 1 implies positive association; < 1 implies negative association."
        )
        generate_log(hyperparams=prb_data, title='Association measure',header=["Metric", "Value"], save_dir=self.save_dir)

        # eta_ = (1/2*(pr_w_thr + pr_kf_thr)) - 1/4
        # ratio_eta = joint_prb / eta_
        # distr_data = {
        #     thr_str:f'({thr_ue},{thr_so})',
        #     str_joint: formatter(joint_prb, digits),
        #     eta_str: formatter(eta_, digits),
        #     r"Ratio": formatter(ratio_eta, digits)
        # }
        # generate_log(hyperparams=distr_data, title='Dominance of the Joint Distribution',header=["Metric", "Value"])
    
    def scatter_kf_workload(self, bs_dict: dict, thr_ue: float, thr_so: float, normalize: bool = False) -> None:
        """Scatter plot of workload vs. covariance norm, with optional normalization."""

        if not bs_dict:
            print("bs_dict is empty. Cannot compute correlation.")
            return
        
        num_points = len(bs_dict)
        cmap = cm.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(num_points)]
        
        plt.figure(figsize=(8, 6))
        w_values, sigma_values = zip(*bs_dict.values())
        
        if normalize:
            interv_w = (np.min(w_values), np.max(w_values))
            interv_sigma = (np.min(sigma_values), np.max(sigma_values))
            w_values = min_max_normalizer(w_values, interv_w)
            sigma_values = min_max_normalizer(sigma_values, interv_sigma)
            thr_ue = min_max_normalizer(thr_ue, interv_w)
            thr_so = min_max_normalizer(thr_so, interv_sigma)
        
        # cnd = len(self.bss) < 20
        #plt.scatter(0,0,marker='.', color='k', label='BS')
        plt.scatter(0,0,marker='o', color='white', edgecolors='k', label='BS')
        
        for (bs, (w, sigma)), color in zip(bs_dict.items(), colors):
            plt.scatter(w, sigma, color=color, edgecolors='k') #label=bs.id if cnd else None
        
        plt.axhline(y=thr_so, color='red', linestyle='--', label=f'KF Threshold ({thr_so})')
        plt.axvline(x=thr_ue, color='k', linestyle=':', label=f'Queue Threshold ({thr_ue})')
        
        # if cnd:
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     unique = dict(zip(labels, handles))
        #     plt.legend(unique.values(), unique.keys(), title='BSs', prop={'size': 8})
        # # else:
        plt.legend()
        
        title_prefix = "Normalized " if normalize else ""
        plt.title(fr"{title_prefix}Queues Workload vs Covariances $l_2$-Norm")
        plt.xlabel(f'{title_prefix}Queues Workload')
        plt.ylabel(fr'{title_prefix}Covariances $l_2$-Norm')
        plt.grid(True)
        plt.show()


    def plot_output(self, thr_ue: float, thr_so: float, steady_state_thr: float):

        tag_map = {
            'jcas': {'so': {'thr': thr_so}, 'ue': {'thr': thr_ue}, 'title': 'JCAS'},
            'sen': {'so': {'thr': thr_so}, 'title': 'Sensing'},
            'com': {'ue': {'thr': thr_ue}, 'title': 'Communication'}
        }.get(self.mode, {})

        if not tag_map:
            raise ValueError(f"Invalid mode: {self.mode}")

        title_label = tag_map.pop('title')
        # num_plots = len(tag_map)

        # fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 3 * num_plots + 3))
        # axs = np.atleast_1d(axs)  # Ensure axs is always iterable

        steady_state_timeline = self.timeline[steady_state_thr:]

        plot_functions = [
            (self.steady_state_empirical_dist, {'steady_state_timeline': steady_state_timeline, 'dtype': 'sinr'}),
            (self.steady_state_empirical_dist, {'steady_state_timeline': steady_state_timeline, 'dtype': 'load'})
        ]

        for func_, extra_args in plot_functions:
            num_plots = len(tag_map)
            fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 3 * num_plots + 3))
            axs = np.atleast_1d(axs)  # Ensure axs is iterable

            for ix, (tag, config) in enumerate(tag_map.items()):
                func_(ax=axs[ix], tag=tag, thr=config['thr'], **extra_args)

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            fig.suptitle(f"{title_label} - Simulation Results", fontsize=18)
            plt.savefig(f"{self.save_dir}/{self.mode}_output_{extra_args['dtype']}.png")
            plt.show()

        # for i, (func_, extra_args) in enumerate(plot_functions):
        #     for ix, (tag, config) in enumerate(tag_map.items()):
        #         func_(ax=axs[ix], tag=tag, thr=config['thr'], **extra_args)

        #     plt.tight_layout(rect=[0, 0, 1, 0.94])
        #     fig.suptitle(f"{title_label} - Simulation Results", fontsize=18)
        #     plt.savefig(f'{self.save_dir}/{self.mode}_output{i}.png')
        #     plt.show()

        

    def steady_state_empirical_dist(self, ax, tag: str, steady_state_timeline, thr: float, dtype: str):
        
        if dtype not in ['sinr', 'load']:
            raise ValueError(f'{dtype} is an invalid *dtype* parameter. Admitted "sinr", "load".')

        tag_map = {
            'so': {
                'load': {
                    'values': lambda so: [matrix_norm(so.kf.covariance_states[t], ord=2) for t in steady_state_timeline],
                    'label': r'KF Error Covariance $\ell_2$-Norm'
                },
                'sinr': {
                    'values': lambda so: [so.sensing_sinrs[t] for t in steady_state_timeline],
                    'label': r'$SINR_{rad}$'
                },
                'set': self.sos
            },
            'ue': {
                'load': {
                    'values': lambda ue: [ue.queue.wloads[t] for t in steady_state_timeline],
                    'label': 'Queue Workload'
                },
                'sinr': {
                    'values': lambda ue: [ue.comm_sinrs[t] for t in steady_state_timeline],
                    'label': r'$SINR_{com}$'
                },
                'set': self.ues
            }
        }

        tag_lower = tag.lower()
        if tag_lower not in tag_map or dtype not in tag_map[tag_lower]:
            raise ValueError('Undefined object or invalid dtype.')

        set_v = tag_map[tag_lower]['set']
        f_v = tag_map[tag_lower][dtype]['values']
        steady_state_v = np.concatenate([f_v(s) for s in set_v])  # More efficient than get_array([...]).flatten()

        label = tag_map[tag_lower][dtype]['label']
        title = label + ' - Steady State Spatial Distribution'
        
        sns.kdeplot(steady_state_v, fill=True, color='orange', ax=ax, label=label)
        if dtype == 'load':
            ax.axvline(x=thr, color='red', linestyle=':', label=f'Threshold ({thr})')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend()


    # def plot_output(self, thr_ue: float, thr_so: float, steady_state_thr: float):

    #     # fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    #     # tag_map = {
    #     #     'so': {'thr': thr_so},
    #     #     'ue': {'thr': thr_ue}
    #     # }

    #     # Determine which tags to plot based on mode
    #     tag_map = {}
    #     title_label = ''
    #     if self.mode == 'jcas':
    #         tag_map = {'so': {'thr': thr_so}, 'ue': {'thr': thr_ue}}
    #         title_label = 'JCAS'
    #         #figsize = (16,9)
    #     elif self.mode == 'sen':
    #         tag_map = {'so': {'thr': thr_so}}
    #         title_label = 'Sensing'
    #         #figsize = (8,6) #(10, 10 / 1.618) #(12,7)
    #     elif self.mode == 'com':
    #         tag_map = {'ue': {'thr': thr_ue}}
    #         title_label = 'Communication'
    #         #figsize = (8,6)#(10, 10 / 1.618)#(12,7)

    #     # Set up figure with correct number of subplots
    #     num_plots = len(tag_map)
    #     fig, axs = plt.subplots(1, num_plots, figsize=(8*num_plots, 3*num_plots+3))
    #     if num_plots == 1:
    #         axs = [axs]  # Ensure axs is iterable when there's only one plot

    #     # Define the functions to call in order
    #     plot_functions = [
    #         (self.steady_state_empirical_dist, {'steady_state_thr': steady_state_thr, 'dtype': 'sinr'}),
    #         (self.steady_state_empirical_dist, {'steady_state_thr': steady_state_thr, 'dtype': 'load'}),
    #         #(self.distribution_temporal_evolution, {})
    #     ]

    #     for func_, extra_args in plot_functions:
    #         for ix, (tag, config) in enumerate(tag_map.items()):
    #             ax = axs[ix]
    #             func_(thr=config['thr'], ax=ax, tag=tag, **extra_args)


    #     plt.tight_layout(rect=[0, 0, 1, 0.94])
    #     fig.suptitle(f"{title_label} - Simulation results", fontsize=18) #Aggregated Analysis of Kalman Filter and Communication Queues
    #     plt.savefig(f'{self.save_dir}/{self.mode}_output.png')
    #     plt.show()

    # def steady_state_empirical_dist(self, ax, tag: str, steady_state_thr: int, thr: float, dtype:str):

    #     steady_state_timeline = self.timeline[steady_state_thr:]
        
    #     if dtype not in ['sinr', 'load']:
    #         raise ValueError('{dtype} is an invalid *dtype* parameter. Admitted "sinr", "load".')
        
    #     if dtype == 'load':

    #         tag_map = {
    #             'so': {
    #                 'values': lambda so: [matrix_norm(so.kf.covariance_states[t], ord=2) for t in steady_state_timeline],
    #                 'label': r'KF Error Covariance $\ell_2$-Norm -',
    #                 'set': self.sos
    #             },
    #             'ue': {
    #                 'values': lambda ue: [ue.queue.wloads[t] for t in steady_state_timeline],
    #                 'label': 'Queue Workload -',
    #                 'set': self.ues
    #             }
    #         }
        
    #     else:

    #         tag_map = {
    #             'so': {
    #                 'values': lambda so: [so.sensing_sinrs[t] for t in steady_state_timeline],
    #                 'label': r'$SINR_{rad}$ -',
    #                 'set': self.sos
    #             },
    #             'ue': {
    #                 'values': lambda ue: [ue.comm_sinrs[t] for t in steady_state_timeline],
    #                 'label': r'$SINR_{com}$ -',
    #                 'set': self.ues
    #             }
    #         }

    #     tag_lower = tag.lower()
    #     if tag_lower not in tag_map:
    #         raise ValueError('Undefined object.')

    #     set_v = tag_map[tag_lower]['set']
    #     f_v = tag_map[tag_lower]['values']
    #     steady_state_v = get_array([f_v(s) for s in set_v]).flatten()
    #     label = tag_map[tag_lower]['label'] + ' Steady State Spatial Distribution'

    #     sns.kdeplot(steady_state_v, fill=True, color='orange', ax=ax, label=label)
    #     ax.axvline(x=thr, color='red', linestyle='--', label=f'Threshold ({thr})')
    #     ax.set_title(label, fontsize=14)
    #     ax.set_xlabel('Values')
    #     ax.set_ylabel('Density')
    #     ax.legend()

    #OLD
    
    def distribution_temporal_evolution(self, thr: float, tag: str, ax):
        tag_map = {
            'so': {
                'values': lambda t, so: matrix_norm(so.kf.covariance_states[t], ord=2),
                'label': 'KL -',
                'set': self.sos
            },
            'ue': {
                'values': lambda t, ue: ue.queue.wloads[t],
                'label': 'Queue -',
                'set': self.ues
            }
        }

        tag_lower = tag.lower()
        if tag_lower not in tag_map:
            raise ValueError('Undefined object.')

        set_v = tag_map[tag_lower]['set']
        f_v = tag_map[tag_lower]['values']
        label = tag_map[tag_lower]['label'] + ' Distribution Temporal Evolution'

        to_plot = [get_array([f_v(t, s) for s in set_v]) for t in self.timeline]

        cmap = plt.get_cmap('viridis', self.t_max)
        step_size = max(1, self.t_max // 10)  # Adjust the step for labels

        for i, t in enumerate(self.timeline):
            values = to_plot[t]
            color = cmap(i / (self.t_max - 1))
            sns.kdeplot(
                values,
                fill=True,
                ax=ax,
                color=color,
                alpha=0.4,
                label=f't = {t}' if i % step_size == 0 else ""
            )

        ax.axvline(x=thr, color='red', linestyle='--', label=f'Threshold ({thr})')
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend()


        # del args_so["lmbda_bss"]

    # def plot_output(self, thr_ue: float, thr_so: float, steady_state_thr: float):
        
    #     fig, axs = plt.subplots(1, 2, figsize=(16, 9)) 
    #     axs = axs.ravel()

    #     # Define plot configurations for each tag
    #     plot_configs = [
    #         {
    #             'tag': 'so',
    #             'thr': thr_so,
    #             'plot_function': self.steady_state_empirical_dist
    #         },
    #         {
    #             'tag': 'ue',
    #             'thr': thr_ue,
    #             'plot_function': self.steady_state_empirical_dist
    #         },
    #         # {
    #         #     'tag': 'so',
    #         #     'label': r'$||\Sigma||_2$',
    #         #     'title': 'Kalman Filter - Distribution Temporal Evolution',
    #         #     'note': r'($||\Sigma_t||_2 = \frac{1}{N} \sum_{i=1}^{N} ||\Sigma_{i,t}||_2$ w. $N = |BSs|$',
    #         #     'note2': r'$||\Sigma||_2 = \{||\Sigma_{i,t}||_2\}_{i=1, t=1}^{N, T}$, $\it{f}\left(\cdot\right) = \it{PDF}$)',
    #         #     'above': kf_above,
    #         #     'thr': thr_so,
    #         #     'plot_function': self.distribution_temporal_evolution
    #         # },
    #         # {
    #         #     'tag': 'ue',
    #         #     'label': r'$||W||_2$',
    #         #     'title': 'Communication Queues - Distribution Temporal Evolution',
    #         #     'note': r'($||W_t||_2 = \frac{1}{N} \sum_{i=1}^{N} ||W_{i,t}||_2$ w. $N = |BSs|$',
    #         #     'note2': r'$||W||_2 = \{||W_{i,t}||_2\}_{i=1, t=1}^{N, T}$, $\it{f}\left(\cdot\right) = \it{PDF}$)',
    #         #     'above': w_above,
    #         #     'thr': thr_ue,
    #         #     'plot_function': self.distribution_temporal_evolution
    #         # }
    #     ]

    #     for ix, config in enumerate(plot_configs):
    #         tag = config['tag']
    #         thr = config['thr']

    #         # Prepare arguments for the plot function
    #         plot_args = {
    #             'thr': thr,
    #             'ax': axs[ix],
    #             'tag': tag
    #         }

    #         if config['plot_function'] == self.steady_state_empirical_dist:
    #             plot_args['steady_state_thr'] = steady_state_thr

    #         # Call the appropriate plotting function with the correct arguments
    #         config['plot_function'](**plot_args)

    #     # Adjust layout to prevent overlap and leave space for titles
    #     plt.tight_layout(rect=[0, 0, 1, 0.94])

    #     # Add a main title for the entire figure
    #     fig.suptitle("Aggregated Analysis of Kalman Filter and Communication Queues", fontsize=18)
        
    #     plt.savefig(f'{self.save_dir}/kfs_queues.png')

    #     plt.show()
    


    # def distribution_temporal_evolution(self, thr:float,tag:str, ax):
    #       # Time steps
    #     to_plot = []
    #     if tag.lower() == 'so':
    #         for t in self.timeline:
    #             # Compute the norm of each base station's covariance matrix at time 't'
    #             covariances = get_array([matrix_norm(so.kf.covariance_states[t], ord=2) for so in self.sos])

    #             to_plot.append(covariances)
    #         # ll = r"$\Sigma'_{i,t}$"
    #         # label = r"$\it{f} \left(\Sigma'_{i,t}\right)$"
    #         label = 'KL -'
    #     elif tag.lower() == 'ue':
    #         for t in self.timeline:
    #             # Compute the norm of each base station's covariance matrix at time 't'
    #             wloads = get_array([ue.queue.wloads[t] for ue in self.ues])

    #             to_plot.append(wloads)
    #         # ll = r"$W'_{i,t}$"
    #         # label = r"$\it{f} \left(W'_{i,t}\right)$"
    #         label = 'Queue -'
    #     else:
    #         raise ValueError('Undefined object.')
        
    #     cmap = plt.get_cmap('viridis', self.t_max)
    #     #s = self.t_max / 10
    #     step_size = max(1, self.t_max // 10)  # Adjust the step for labels
    #     #flattened_values = flatten(to_plot)
    #     #clipped_min, clipped_max = np.percentile(flattened_values, [1, 99])  # Clip extreme values
    #     for i, t in enumerate(self.timeline):
    #         # values = np.clip(to_plot[t], clipped_min, clipped_max)  # Clip KDE input
    #         values = to_plot[t]  # Clip KDE input
    #         color = cmap(i / (self.t_max - 1))
    #         sns.kdeplot(
    #             values, 
    #             fill=True, 
    #             ax=ax, 
    #             color=color, 
    #             alpha=0.4, 
    #             label=f't = {t}' if i % step_size == 0 else ""
    #         )
    #     # for i,t in enumerate(self.timeline):
    #     #     color = cmap(i / (self.t_max - 1))
    #     #     sns.kdeplot(to_plot[t], fill=True, ax=ax, color=color, alpha=0.5, label=f't = {t}' if i % s == 0 else "")
        
    #     ax.axvline(x=thr, color='red', linestyle='--', label=f'Threshold ({thr})')
    #     # Update labels and title
    #     ax.set_title(f'{label} Distribution Temporal Evolution')
    #     ax.set_xlabel('Values')
    #     ax.set_ylabel('Density')
    #     #ax.set_xlim(clipped_min, clipped_max)  # Apply clipping to the plot
    #     ax.legend()
    #     #ax.set_xlim(right=0.8*max_v) #left=0.8*min_v
    # def plot_output(self, thr_ue: float, thr_so: float, steady_state_thr: float):
    #     fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    #     axs = axs.ravel()

    #     tag_map = {
    #         'so': {
    #             'thr': thr_so,
    #             'plot_function1': self.steady_state_empirical_dist,
    #             'plot_function2': self.distribution_temporal_evolution,
    #         },
    #         'ue': {
    #             'thr': thr_ue,
    #             'plot_function1': self.steady_state_empirical_dist,
    #             'plot_function2': self.distribution_temporal_evolution,
    #         }
    #     }

    #     for ix, (tag, config) in enumerate(tag_map.items()):
    #         plot_args = {
    #             'thr': config['thr'],
    #             'ax': axs[ix],
    #             'tag': tag
    #         }
            
    #         if config['plot_function'] == self.steady_state_empirical_dist:
    #             plot_args['steady_state_thr'] = steady_state_thr
            
    #         config['plot_function'](**plot_args)

    #     plt.tight_layout(rect=[0, 0, 1, 0.94])
    #     fig.suptitle("Aggregated Analysis of Kalman Filter and Communication Queues", fontsize=18)
    #     plt.savefig(f'{self.save_dir}/kfs_queues.png')
    #     plt.show()

    
    # def steady_state_empirical_dist(self,  ax, tag:str, steady_state_thr:int, thr:float):

    #     steady_state_v = []
    #     steady_state_timeline = self.timeline[steady_state_thr:]

    #     if tag.lower() == 'so':
    #         steady_state_v = get_array([([matrix_norm(bs.kfs[0].covariance_states[t], ord=2) for t in steady_state_timeline]) for bs in self.bss]).flatten()
    #         label = 'KF -'
    #     elif tag.lower() == 'ue':
    #         steady_state_v = get_array([([bs.queues[0].queue.wloads[t] for t in steady_state_timeline]) for bs in self.bss]).flatten()
    #         label = 'Queue -'
    #     else:
    #         raise ValueError('Undefined object.')
        
    #     label += ' Steady State Spatial Distribution' 
    #     sns.kdeplot(steady_state_v, fill=True, color='orange',ax=ax,label=label)
    #     ax.axvline(x=thr, color='red', linestyle='--', label=f'Threshold ({thr})')
    #     ax.set_title(label, fontsize=14)
    #     ax.set_xlabel('Values')
    #     ax.set_ylabel('Density')
    #     ax.legend()

            
    
    # def run(self, velox_domain: list, thr_ue:float,thr_so:float, kf_params:dict ,
    #         arrival_rate_domain:list,transl_scaler_domain:list,
    #          show_plot:bool=False,show_samples:bool=False,
    #          tail:bool = False, kf_meas_dim:int=2):
    #     """
    #     Runs the simulation for the base station and objects.

    #     Parameters:
    #     objs (list): List of `Obj` instances to track as sources of signals.
    #     velox_domain (list): Range for setting velocities of the objects.
    #     neighbors (list): List of neighboring base stations.
    #     n_steps (int): Number of steps for Kalman filter simulation.
    #     """

    #     self.netManager = NetManager(bss:list, ues:list, sos:list, n_steps:int, show_plot: bool = False)

    #     self.netManager.start(args_so=args_so, args_ue=args_ue)

    #     self.generate_env_description()

    #     del args_so["lmbda_bss"]

    #     self.netManager.run()

    #     steady_state_thr = round(0.3*self.t_max)
    #     self.plot_output(thr_ue = thr_ue,thr_so = thr_so,
    #                                  show_plot=show_plot,steady_state_thr=steady_state_thr)
    #     self.measure_association(thr_ue = thr_ue,thr_so = thr_so,steady_state_thr = steady_state_thr,
    #                         tail=tail)
    #     self.steady_state_correlation(steady_state_thr=steady_state_thr, thr_ue=thr_ue,thr_so=thr_so)

        

    #     # Step 1: Initialize objects and run their processes
    #     for so in self.sos:
    #         so.start(velox_domain=velox_domain)
            
    #     for ue in self.ues:
    #         ue.start(velox_domain=velox_domain)

    #     cond1 = isinstance(arrival_rate_domain, list) and len(arrival_rate_domain) >= 2
    #     cond2 = isinstance(transl_scaler_domain, list) and len(transl_scaler_domain) >= 2

    #     if not cond1 and isinstance(arrival_rate_domain, list):
    #         arrival_rate_domain = arrival_rate_domain[0]
        
    #     if not cond2 and isinstance(transl_scaler_domain, list):
    #         transl_scaler_domain = transl_scaler_domain[0]

    #     cond_bss = len(self.bss) <= 50
        

    #     if cond1 or cond2:
    #         vc_diam_dict = {bs: get_vc_diam(bs.region) for bs in self.bss}
            
    #         # 3) Sort by the vc_diam value (the second item in the (bs.id, vc_diam) pair)
    #         sorted_vc = sorted(vc_diam_dict.items(), key=lambda x: x[1])  # ascending order by vc_diam
            
    #         if cond1:
    #             arrival_rates = np.linspace(min(arrival_rate_domain), max(arrival_rate_domain), len(self.bss), dtype=int)
    #         else:
    #             arrival_rates = arrival_rate_domain

    #         if cond2:
    #             trans_matrix_scaler = np.linspace(min(transl_scaler_domain), max(transl_scaler_domain), len(self.bss))
    #         else:
    #             trans_matrix_scaler = transl_scaler_domain
            
    #         for i, (bs, _) in enumerate(sorted_vc):            
    #             bs.set_arrival_rate(arrival_rates[i])         
    #             bs.set_transition_scaler(trans_matrix_scaler[i]) 
    #             bs_neighbors = [other_bs for other_bs in self.bss if other_bs != bs]
    #             bs.set_neighbors_bss(bs_neighbors)
    #             if cond_bss:
    #                 bs_params[bs.id] = r'$\lambda_{arr}$' + f'= {formatter(arrival_rates[i], digits)}, '+r'$\rho$'+ f'= {formatter(trans_matrix_scaler[i],digits)}'
    #     else:
    #         for bs in self.bss:            
    #             bs.set_arrival_rate(arrival_rate_domain)         
    #             bs.set_transition_scaler(transl_scaler_domain) 
    #             bs_neighbors = [other_bs for other_bs in self.bss if other_bs != bs]
    #             bs.set_neighbors_bss(bs_neighbors)
    #             # if cond_bss:
    #             #     bs_params[bs.id] = r'$\lambda_{arr}$' + f'= {formatter(arrival_rate_domain, digits)}, '+r'$\rho_{mov}$'+ f'= {formatter(transl_scaler_domain,digits)}'
            
        
    #     for bs in self.bss:
    #         bs.run(n_steps = self.t_max, show_plot = show_samples,kf_thr=thr_so, w_thr=thr_ue,
    #                 kf_params = kf_params, kf_meas_dim = kf_meas_dim)
            
        
    # def steady_state_correlation(self, steady_state_thr:int, thr_ue:float,thr_so:float):

    #     steady_state_timeline = self.timeline[steady_state_thr:]

    #     avg_covariances = get_array(
    #         [
    #         get_temporal_avg([matrix_norm(bs.kfs[0].covariance_states[t], ord=2) for t in steady_state_timeline], steady_state_thr) 
    #         for bs in self.bss
    #         ])
        
    #     avg_queue_lengths = get_array([
    #         get_temporal_avg([bs.queues[0].queue.wloads[t] for t in steady_state_timeline],steady_state_thr) 
    #         for bs in self.bss
    #         ])
            
    #     bs_dict = {}
    #     for bs, w, sigma in zip(self.bss, avg_queue_lengths, avg_covariances):
    #         bs_dict[bs] = (w, sigma)
        
    #     self.scatter_kf_workload(bs_dict = bs_dict, thr_ue=thr_ue,thr_so=thr_so)
        

    # def scatter_kf_workload_n(self, bs_dict:dict, thr_ue:float,thr_so:float):

    #     if not bs_dict:
    #         print("bs_dict is empty. Cannot compute correlation.")
    #         return

    #     num_points = len(bs_dict)
    
    #     # Generate a list of distinct colors using a colormap
    #     cmap = cm.get_cmap('tab20')  # 'tab20' has 20 distinct colors
    #     colors = [cmap(i % 20) for i in range(num_points)]
        
    #     plt.figure(figsize=(8, 6))
        
    #     # Plot each point with a unique color
    #     w_values, sigma_values = zip(*bs_dict.values())
        
    #     interv_w = (np.min(w_values), np.max(w_values))
    #     interv_sigma = (np.min(sigma_values), np.max(sigma_values))

    #     w_values_norm = min_max_normalizer(w_values,interv_w)
    #     sigma_values_norm = min_max_normalizer(sigma_values,interv_sigma)
    #     thr_ue = min_max_normalizer(thr_ue,interv_w)
    #     thr_so = min_max_normalizer(thr_so,interv_sigma)

    #     for bs, w, sigma, color in zip(self.bss,w_values_norm,sigma_values_norm, colors):
    #         plt.scatter(w, sigma, color=color, edgecolors='k', label=bs.id)
        
    #     # To avoid duplicate labels in the legend, create a legend with unique labels
    #     plt.axhline(y=thr_so, color='red', linestyle='--', label=f'Normalized KF Threshold')
    #     plt.axvline(x=thr_ue, color='k', linestyle=':', label=f'Normalized Queue Threshold')

    #     if len(self.bss) < 20:
    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         unique = dict(zip(labels, handles))
    #         plt.legend(unique.values(), unique.keys(), title='BSs', prop={'size': 8}) #

        
    #     # plt.colorbar(scatter, label='BSs')
    #     #plt.title(f"Pearson Correlation coefficient: {formatter(correlation, digits)}")
    #     plt.title(r"Normalized Queues Workload vs Covariances $l_2$-Norm")
    #     plt.xlabel('Normalized Queues Workload')
    #     plt.ylabel(r'Normalized Covariances $l_2$-Norm')
    #     plt.grid(True)
    #    #plt.legend(title='BSs') #prop={'size': 6}
    #     plt.show()


    # def scatter_kf_workload(self, bs_dict:dict, thr_ue:float,thr_so:float):

    #     if not bs_dict:
    #         print("bs_dict is empty. Cannot compute correlation.")
    #         return
        
    #     # # Extract separate lists for w and sigma
    #     # try:
    #     #     w_values, sigma_values = zip(*bs_dict.values())
    #     # except ValueError:
    #     #     print("bs_dict does not contain valid (w, sigma) tuples.")
    #     #     return
        
    #     # if len(w_values) < 2:
    #     #     print("Not enough data points to compute correlation.")
    #     #     return
        
    #     # Compute Pearson correlation coefficient
    #     # digits = 3
    #     # correlation = np.corrcoef(w_values, sigma_values)[0, 1]
    #     # print(f"Pearson correlation coefficient: {formatter(correlation, digits)}")

    #     num_points = len(bs_dict)
    
    #     # Generate a list of distinct colors using a colormap
    #     cmap = cm.get_cmap('tab20')  # 'tab20' has 20 distinct colors
    #     colors = [cmap(i % 20) for i in range(num_points)]
        
    #     plt.figure(figsize=(8, 6))

    #     cnd = len(self.bss) < 20
        
    #     # Plot each point with a unique color
    #     for (bs, (w, sigma)), color in zip(bs_dict.items(), colors):
    #         if cnd:
    #             plt.scatter(w, sigma, color=color, edgecolors='k', label=bs.id)
    #         else:
    #             plt.scatter(w, sigma, color=color, edgecolors='k')
        
    #     # To avoid duplicate labels in the legend, create a legend with unique labels
    #     plt.axhline(y=thr_so, color='red', linestyle='--', label=f'KF Threshold ({thr_so})')
    #     plt.axvline(x=thr_ue, color='k', linestyle=':', label=f'Queue Threshold ({thr_ue})')

    #     if cnd:
    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         unique = dict(zip(labels, handles))
    #         plt.legend(unique.values(), unique.keys(), title='BSs', prop={'size': 8})
    #     else:
    #         plt.legend() #

        
    #     # plt.colorbar(scatter, label='BSs')
    #     #plt.title(f"Pearson Correlation coefficient: {formatter(correlation, digits)}")
    #     plt.title(r"Queues Workload vs Covariances $l_2$-Norm")
    #     plt.xlabel('Queues Workload')
    #     plt.ylabel(r'Covariances $l_2$-Norm')
    #     plt.grid(True)
    #    #plt.legend(title='BSs') #prop={'size': 6}
    #     plt.show()
        

        
    

    


    

        