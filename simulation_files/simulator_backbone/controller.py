from simulation_files.simulator_backbone.network_instances import * 
import os
import threading
from concurrent.futures import wait, ThreadPoolExecutor, ProcessPoolExecutor
import time

class NetManager:

    def __init__(self, bss:list, ues:list, sos:list, n_steps:int, verbose:bool, show_samples: bool, save_dir:str, mode:str):
        self.bss = bss
        self.ues = ues
        self.sos = sos
        self.n_steps = n_steps 
        self.verbose = verbose
        self.show_samples = show_samples
        #self.manager = multiprocessing.Manager()
        # self.lock = self.manager.Lock()
        #self.lock = threading.Lock()
        self.save_dir = save_dir
        self.mode = mode

        self._stop_event = threading.Event()
        self._beamformer_thread = None
    
    def _move_reference_vectors_continuously(self, interval=0.5):
        """Continuously moves reference vectors at fixed time intervals."""
        while not self._stop_event.is_set():
            for bs in self.bss:
                bs.beamformer.move_reference_vector()
            time.sleep(interval)  # Tune this to your desired update rate

    def start_reference_vector_thread(self, interval=0.5):
        """Starts background thread for continuous reference vector movement."""
        self._beamformer_thread = threading.Thread(
            target=self._move_reference_vectors_continuously, 
            args=(interval,),
            daemon=True
        )
        self._beamformer_thread.start()

    def stop_reference_vector_thread(self):
        """Signals thread to stop and waits for it to exit."""
        self._stop_event.set()
        if self._beamformer_thread is not None:
            self._beamformer_thread.join()
        
    def _update_steady_state(self, bs):
        ues_ = lookup_bs(items=self.ues, id=bs.id)
        sos_ = lookup_bs(items=self.sos, id=bs.id)

        w = np.mean(get_array([ue.queue.wloads[-1] for ue in ues_]))
        sigma = np.mean(get_array([matrix_norm(so.kf.covariance_states[-1], ord=2) for so in sos_]))

        bs.steady_state_kf_wload.append((w, sigma))
    
    # Ensure _update_so returns updated so
    def _update_so(self, so):
        so.update_estimate()
        so.update_parameters()
        return so

    # Wrapper for ue.update_queue to return ue after update
    def _update_ue(self, ue, t):
        ue.update_queue(t)
        return ue

    def assign_to_bs(self, obj):
        """Assigns a UE or SO to the closest BS based on torus distance."""
        return min(self.bss, key=lambda bs: torus_distance(x=bs.position, y=obj.position))

    def get_handover_summary(self):
        return {
            k: {tag: [obj.id for obj in lst] for tag, lst in dc.items()}
            for k, dc in self.assignment_matrix.items()
        }
    
    def handover(self, obj:Obj, tag:str, log_filename:str): #either ues or sos
        """Handles UE or SO handover in a thread-safe manner."""
        #with self.lock:
        tag = tag.lower()
        assert tag in ["ues", "sos"], f"Invalid tag: {tag}"

        old_bs = obj.bs
        new_bs = self.assign_to_bs(obj)
        check_flag = (new_bs.id != old_bs.id) and (old_bs is not None)

        if check_flag:
            with self.lock:
                self.assignment_matrix[old_bs.id][tag].remove(obj)
                self.assignment_matrix[new_bs.id][tag].append(obj)
                obj.bs = new_bs
                obj.region = new_bs.region

                if self.verbose and (log_filename is not None):
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"OBJ ID: {obj.id}, Former BS ID: {old_bs.id}\n")
                        log_file.write(f"OBJ ID: {obj.id}, Assigned BS ID: {new_bs.id}\n")

                #if you want to update the handover_log
                #obj.handover_log.append(new_bs.id)

                #redundat
                # if tag == 'ues':
                #     old_bs.ues_.remove(obj)                            
                #     new_bs.ues_.append(obj)
                # else:
                #     print('BS ID: ', old_bs.id, ', SO IDs:', [so.id for so in old_bs.sos_])
                #     print('SO ID to be removed: ', obj.id)
                #     old_bs.sos_.remove(obj)                            
                #     new_bs.sos_.append(obj)

    def start(self, args_so: dict, args_ue: dict):
        """Starts all SOs and UEs with given arguments."""
        for bs in self.bss:
            bs_neighbors = [other_bs for other_bs in self.bss if other_bs != bs]
            bs.start(neighbors = bs_neighbors, mode = self.mode)

        for so in self.sos:
            so.start(args=args_so)

        for ue in self.ues:
            ue.start(args=args_ue)
                
        self.assignment_matrix = {bs.id: {"ues": bs.ues_, "sos": bs.sos_} for bs in self.bss} #self.manager.dict()

        self.ues_moving = [ue for ue in self.ues if ue.motion]
        self.sos_moving = [so for so in self.sos if so.motion]                                 

    def network_subroutine(self, t:int, flag:bool, log_filename:str, thread_executor: ThreadPoolExecutor, process_executor: ProcessPoolExecutor):

             #everything has to be parallelized

        #@MARK: alternative
        # for ue in self.ues_moving:
        #     ue.move(t=t)
        # for so in self.sos_moving: #sos_moving
        #     so.move()  

        #then handover
        #@MARK: alternative
        # for ue in self.ues_moving:
        #     self.handover(obj=ue, tag='ues', log_filename=log_filename)
        #SO not subjected to handovwr
        # for so in self.sos_moving:
        #     self.handover(obj=so, tag='sos', log_filename=log_filename)

        # for so in self.sos:
        #     so.update_estimate()
        #     so.update_parameters()

        # for ue in self.ues:
        #     ue.update_queue(t=t)

        # if self.verbose and (log_filename is not None):
        #     with open(log_filename, 'a') as log_file:
        #         log_file.write(f"Time: {t} - Handovers Logs\n")

        
        # motion
        futures_motion = [thread_executor.submit(ue.move, t) for ue in self.ues_moving] + \
                        [thread_executor.submit(so.move) for so in self.sos_moving]
        # wait for motion done
        wait(futures_motion)

        if self.verbose and (log_filename is not None):
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Time: {t} - Handovers Logs\n")

        # handover
        futures_handover = [thread_executor.submit(self.handover, ue, 'ues', log_filename) for ue in self.ues_moving]
        wait(futures_handover)

        # Step 4: Updates
        # futures_update = [thread_executor.submit(self._update_so, so) for so in self.sos] + \
        #                 [thread_executor.submit(ue.update_queue, t) for ue in self.ues]
        # wait(futures_update)
        futures_update_so = [thread_executor.submit(self._update_so, so=so) for so in self.sos]
        futures_update_ue = [thread_executor.submit(self._update_ue, ue=ue, t=t) for ue in self.ues]

        wait(futures_update_so + futures_update_ue)

        # Collect results
        updated_sos = [f.result() for f in futures_update_so]
        updated_ues = [f.result() for f in futures_update_ue]

        # Update the main lists
        self.sos = updated_sos
        self.ues = updated_ues
        
        # steady-state update if flag is set
        if flag:
            futures_steady = [thread_executor.submit(self._update_steady_state, bs) for bs in self.bss]
            wait(futures_steady)
    
        if self.verbose and (log_filename is not None):
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Time: {t} - Network Snapshot\n")

                # Log base station details
                for bs in self.bss:
                    log_file.write(f"BS ID: {bs.id}\n")
                    log_file.write(f"UEs ID: {[ue.id for ue in bs.ues_]}\n")
                    log_file.write(f"SOs ID: {[so.id for so in bs.sos_]}\n")
                    log_file.write('\n')


        # for k, dc in self.assignment_matrix.items():
        #     print(k, {tag: [obj.id for obj in lst] for tag, lst in dc.items()})
        # print(self.get_handover_summary())
        

        
        # with ThreadPoolExecutor() as executor:
        #     executor.map(lambda ue: self.handover(obj=ue, tag='ues', log_filename=log_filename), self.ues_moving)
            #executor.map(lambda so: self.handover(obj=so, tag='sos', log_filename=log_filename), self.sos_moving)
        

        #then update estimators
        # with ThreadPoolExecutor() as executor:
        #     futures = []
        #     futures.extend(executor.submit(so.update_estimate) for so in self.sos)
        #     futures.extend(executor.submit(so.update_parameters) for so in self.sos)
        #     futures.extend(executor.submit(ue.update_queue, t) for ue in self.ues)
        #     for f in futures:
        #         f.result()
        
        # for bs in self.bss:
        #     bs.beamformer.move_reference_vector()
        
        #bs collect the wload and kf
        # if flag: #if there is only one so and one ue the the average will give allways the same value
        #     for bs in self.bss:
        #         ues_ = lookup_bs(items=self.ues, id=bs.id)
        #         sos_ = lookup_bs(items=self.sos, id=bs.id)

        #         w = np.mean(get_array([ue.queue.wloads[-1] for ue in ues_]))
        #         sigma = np.mean(get_array([matrix_norm(so.kf.covariance_states[-1], ord=2) for so in sos_]))

        #         bs.steady_state_kf_wload.append((w, sigma))

    
    
    def run(self, steady_state_thr:int):
        """Runs the network simulation for n_steps with a progress bar."""

        log_filename = None
        if self.verbose:
            os.makedirs(self.save_dir, exist_ok=True)
            log_filename = os.path.join(self.save_dir, "log_file.txt")
        # Open the log file in append mode
        #flag = False
        self.start_reference_vector_thread(interval=0.1)
        with ThreadPoolExecutor(max_workers=1) as thread_executor, ProcessPoolExecutor(max_workers=1) as process_executor: #os.cpu_count()
            for t in tqdm(range(self.n_steps), desc="Running simulation"):
            # for t in range(self.n_steps):
                flag = t >= steady_state_thr
                self.network_subroutine(t=t, flag=flag, log_filename=log_filename, 
                                        thread_executor=thread_executor,process_executor=process_executor)

            self.stop_reference_vector_thread()

            futures_finish = [process_executor.submit(ue.finish, show_samples=self.show_samples) for ue in self.ues] + \
                         [process_executor.submit(so.finish, show_samples=self.show_samples) for so in self.sos]
            
            wait(futures_finish)
        #flag = False
        # for ue in self.ues:
        #     ue.finish(show_samples = self.show_samples)

        # for so in self.sos:
        #     so.finish(show_samples = self.show_samples)

        if self.show_samples:
            self.aggregate_plots()


    def aggregate_plots(self) -> None:
        """
        Aggregate and display plots for all Kalman Filters and Queues.
        """
        dir_ = f'{self.save_dir}/BSs'
        os.makedirs(dir_, exist_ok=True)

        def plot_set(objs, plot_list, axs, plot_idx):
            """Helper function to iterate over objects and plot functions."""
            for obj in objs:
                for plot_func_name, args in plot_list:
                    plot_func = getattr(obj.plotter, plot_func_name)
                    plot_args = {'ax': axs[plot_idx]}
                    if args is not None:
                        plot_args.update(args)
                    plot_func(plot_args)
                    plot_idx += 1
            return plot_idx

        for bs in self.bss:
            # Define plot configurations for Kalman Filters and Queues
            kf_plots = [
                ("kf_evolution_1D", {"coord": "x"}), ("kf_evolution_1D", {"coord": "y"}),
                ("kf_evolution_1D", {"coord": "v_x"}), ("kf_evolution_1D", {"coord": "v_y"}),
                ("stability_checker", None), ("kf_error_process", None),
                ("covariance_evolution", None), ("sinr_evolution", None)
            ]
            queue_plots = [
                ("queue_evolution", None), ("sinr_evolution", None),
                ("lindley_evolution", None)
            ]

            n_kfs, n_queues = len(bs.sos_), len(bs.ues_)
            total_plots = n_kfs * len(kf_plots) + n_queues * len(queue_plots)

            # Define subplot grid
            n_cols = 2
            n_rows = max(1, ceil(total_plots / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))
            axs = [axs] if total_plots == 1 else axs.flatten()

            plot_idx = 0  # Initialize plot index

            # Plot Kalman Filters and Queues
            if self.mode != 'com':
                plot_idx = plot_set(bs.sos_, kf_plots, axs, plot_idx)
                
            if self.mode != 'sen':
                plot_idx = plot_set(bs.ues_, queue_plots, axs, plot_idx)

            # Remove unused subplots
            for i in range(plot_idx, len(axs)):
                fig.delaxes(axs[i])

            # Add title and adjust layout
            fig.suptitle(f"{bs.id.upper()} \n Kalman Filter and Queue Simulation Results", fontsize=20)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"{dir_}/{bs.id}.png")
            plt.show()

    # def aggregate_plots(self) -> None:
    #     """
    #     Aggregate and display plots for all Kalman Filters and Queues, saving each plot as a separate image.
    #     """
    #     dir_ = f'{self.save_dir}/BSs'
    #     os.makedirs(dir_, exist_ok=True)

    #     def save_single_plot(obj, plot_func_name, args, bs_id, plot_type, suffix=""):
    #         """Helper function to create and save a single plot."""
    #         fig, ax = plt.subplots(figsize=(7, 6))
    #         plot_args = {'ax': ax}
    #         if args is not None:
    #             plot_args.update(args)
                
    #         plot_func = getattr(obj.plotter, plot_func_name)
    #         plot_func(plot_args)
            
    #         # Create a descriptive filename
    #         if hasattr(obj, 'id'):
    #             obj_id = obj.id
    #         else:
    #             obj_id = f"{plot_type}_{obj.__class__.__name__}"
            
    #         new_dir = f'{dir_}/{bs_id}'
    #         os.makedirs(new_dir, exist_ok=True)
                
    #         filename = f"{new_dir}/{obj_id}_{plot_func_name}{suffix}.png"
    #         plt.tight_layout()
    #         plt.savefig(filename)
    #         plt.show()
    #         plt.close(fig)

    #     for bs in self.bss:
    #         # Define plot configurations for Kalman Filters and Queues
    #         kf_plots = [
    #             ("kf_evolution_1D", {"coord": "x"}), ("kf_evolution_1D", {"coord": "y"}),
    #             ("kf_evolution_1D", {"coord": "v_x"}), ("kf_evolution_1D", {"coord": "v_y"}),
    #             ("stability_checker", None), ("kf_error_process", None),
    #             ("covariance_evolution", None), ("sinr_evolution", None)
    #         ]
    #         queue_plots = [
    #             ("queue_evolution", None), ("sinr_evolution", None),
    #             ("lindley_evolution", None)
    #         ]

    #         # Plot Kalman Filters
    #         if self.mode != 'com':
    #             for i, so in enumerate(bs.sos_):
    #                 for plot_func_name, args in kf_plots:
    #                     suffix = f"_{args['coord']}" if args and 'coord' in args else ""
    #                     save_single_plot(so, plot_func_name, args, bs.id, "KF", suffix)
                        
    #         # Plot Queues
    #         if self.mode != 'sen':
    #             for i, ue in enumerate(bs.ues_):
    #                 for plot_func_name, args in queue_plots:
    #                     save_single_plot(ue, plot_func_name, args, bs.id, "Queue")
    

    #@OLD


    # def aggregate_plots(self) -> None:
    #     """
    #     Aggregate and display plots for all Kalman Filters and Queues.
    #     """
        
    #     # def plot(plot_func, ax, args):
    #     #     """Helper function to plot if there's an available subplot."""
    #     #     if arg is not None:
    #     #         plot_func(ax=ax, *args)
    #     #     else:
    #     #         plot_func(ax=ax)

    #     # def plot_if_available(plot_func, obj, *args):
    #     #     """Helper function to plot if there's an available subplot."""
    #     #     nonlocal plot_idx
    #     #     if plot_idx < len(axs):
    #     #         if args:  # Ensure args is only passed when it exists
    #     #             plot_func(ax=axs[plot_idx], *args)
    #     #         else:
    #     #             plot_func(ax=axs[plot_idx])
    #     #         plot_idx += 1
    #     def plot_set(objs,plot_list, axs, plot_idx):
    #         for obj in objs:
    #             for plot_func_name, args in plot_list:
    #                 plot_func = getattr(obj.plotter, plot_func_name)
    #                 if args is not None:
    #                     plot_func(ax=axs[plot_idx], *args)
    #                 else:
    #                     plot_func(ax=axs[plot_idx])
    #                 # plot(plot_func=plot_func, ax=axs[plot_idx], args=args)
    #                 plot_idx +=1

    #         return plot_idx


    #     for bs in self.bss:

    #         # Plot configurations for Kalman Filters
    #         kf_plots = [
    #             ("kf_evolution_1D", "x"), ("kf_evolution_1D", "y"),
    #             ("kf_evolution_1D", "v_x"), ("kf_evolution_1D", "v_y"),
    #             ("stability_checker", None), ("kf_error_process", None),
    #             ("covariance_evolution", None)
    #         ]

    #         # Plot configurations for Queues
    #         queue_plots = [
    #             ("queue_evolution", None), ("shannon_rate_evolution", None),
    #             ("lindley_evolution", None)
    #         ]


    #         n_kfs, n_queues = len(bs.sos_), len(bs.ues_)
    #         plots_per_kf, plots_per_queue = len(kf_plots), len(queue_plots)  # Now considering Lindley evolution
    #         total_plots = n_kfs * plots_per_kf + n_queues * plots_per_queue

    #         # Define subplot grid size (e.g., 2 columns)
    #         n_cols = 2
    #         n_rows = max(1, ceil(total_plots / n_cols))

    #         # Create subplots
    #         fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))
    #         axs = [axs] if total_plots == 1 else axs.flatten()
            
    #         plot_idx = 0  # Initialize plot index

    #         # Plot Kalman Filters
    #         plot_idx = plot_set(objs = bs.sos_, plot_list = kf_plots, axs=axs, plot_idx= plot_idx)
    #         plot_idx = plot_set(objs = bs.ues_, plot_list = queue_plots, axs=axs, plot_idx= plot_idx)

    #         # for so in bs.sos_:
    #         #     for plot_func_name, args in kf_plots:
    #         #         plot_func = getattr(so.plotter, plot_func_name)
    #         #         # if arg is not None:
    #         #         #     plot_func(ax=axs[plot_idx], *args)
    #         #         # else:
    #         #         #     plot_func(ax=axs[plot_idx])
    #         #         plot(plot_func=plot_func, ax=axs[plot_idx], args=args)
    #         #         plot_idx +=1

    #         # Plot Queues
    #         # for ue in bs.ues_:
    #         #     for plot_func_name, arg in queue_plots:
    #         #         plot_func = getattr(ue.plotter, plot_func_name)
    #         #         # if arg is not None:
    #         #         #     plot_func(ax=axs[plot_idx], *args)
    #         #         # else:
    #         #         #     plot_func(ax=axs[plot_idx])
    #         #         plot(plot_func=plot_func, ax=axs[plot_idx], args=args)
    #         #         plot_idx +=1

    #         # Remove unused subplots
    #         for i in range(plot_idx, len(axs)):
    #             fig.delaxes(axs[i])

    #         # Add title and adjust layout
    #         fig.suptitle(f"{bs.id.upper()} \n Aggregated Kalman Filter and Queue Simulation Results", fontsize=20)
    #         plt.tight_layout(rect=[0, 0, 1, 0.97])
            
    #         # Display the figure
    #         plt.show()
#%%%%%%%%%%%
            
    
    # def aggregate_plots(self) -> None:
    #     """
    #     Aggregate and display plots for all Kalman Filters and Queues.
    #     """
    #     for bs in self.bss:
    #         n_kfs = len(bs.sos_)
    #         n_queues = len(bs.ues_)

    #         plots_per_kf = 7# if not DEBUG else 5
    #         plots_per_queue = 2

    #         total_kf_plots = n_kfs * plots_per_kf
    #         total_queue_plots = n_queues * plots_per_queue
    #         total_plots = total_kf_plots + total_queue_plots

    #         # Define subplot grid size (e.g., 3 columns)
    #         n_cols = 2
    #         n_rows = ceil(total_plots / n_cols) if total_plots > 0 else 1

    #         # Create subplots
    #         fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))
    #         if total_plots == 1:
    #             axs = [axs]  # Make it iterable
    #         else:
    #             axs = axs.flatten()

    #         plot_idx = 0  # Initialize plot index

    #         if len(bs.sos_) >0:
    #             for so in bs.sos_:

    #                 # Plot Kalman Filter Position Evolution
    #                 if plot_idx < len(axs):
    #                     so.plotter.kf_evolution_1D(ax=axs[plot_idx], coord='x')
    #                     plot_idx += 1
                    
    #                 if plot_idx < len(axs):
    #                     so.plotter.kf_evolution_1D(ax=axs[plot_idx], coord='y')
    #                     plot_idx += 1

    #                 if plot_idx < len(axs):
    #                     so.plotter.kf_evolution_1D(ax=axs[plot_idx], coord='v_x')
    #                     plot_idx += 1

    #                 if plot_idx < len(axs):
    #                     so.plotter.kf_evolution_1D(ax=axs[plot_idx], coord='v_y')
    #                     plot_idx += 1

    #                 #if DEBUG:
    #                 if plot_idx < len(axs):
    #                     so.plotter.stability_checker(ax=axs[plot_idx])
    #                     #axs[plot_idx].set_title(f"{self.id.upper()} - Stability Checker")
    #                     plot_idx += 1


    #                 if plot_idx < len(axs):
    #                     so.plotter.kf_error_process(ax=axs[plot_idx])
    #                     #axs[plot_idx].set_title(f"{self.id.upper()} - Stability Checker")
    #                     plot_idx += 1

    #                 # Plot Covariance Distribution
    #                 # if plot_idx < len(axs):
    #                 #     so.plotter.covariance_distribution(ax=axs[plot_idx])
    #                 #     #axs[plot_idx].set_title(f"{self.id.upper()} - Covariance Distribution")
    #                 #     plot_idx += 1

    #                 # Plot Covariance Evolution
    #                 if plot_idx < len(axs):
    #                     so.plotter.covariance_evolution(ax=axs[plot_idx])
    #                     #axs[plot_idx].set_title(f"{self.id.upper()} - Covariance Evolution")
    #                     plot_idx += 1

    #         if len(bs.ues_) >0:
    #             for ue in bs.ues_:
    #                 # Plot Queue Evolution
    #                 if plot_idx < len(axs):
    #                     ue.plotter.queue_evolution(ax=axs[plot_idx])
    #                     #axs[plot_idx].set_title(f"{self.id.upper()} - Queue Evolution")
    #                     plot_idx += 1

    #                 # Plot Shannon Rate Evolution
    #                 if plot_idx < len(axs):
    #                     ue.plotter.shannon_rate_evolution(ax=axs[plot_idx])
    #                     #axs[plot_idx].set_title(f"{self.id.upper()} - Shannon Rate Evolution")
    #                     plot_idx += 1

    #                 if plot_idx < len(axs):
    #                     ue.plotter.lindley_evolution(ax=axs[plot_idx])
    #                     #axs[plot_idx].set_title(f"{self.id.upper()} - Queue Evolution")
    #                     plot_idx += 1

    #                 # if plot_idx < len(axs):
    #                 #     so.plotter.local_correlation(wloads=ue.queue.wloads, ax=axs[plot_idx])
    #                 #     #axs[plot_idx].set_title(f"{self.id.upper()} - Shannon Rate Evolution")
    #                 #     plot_idx += 1
                

    #         # Remove any unused subplots
    #         for i in range(plot_idx, len(axs)):
    #             fig.delaxes(axs[i])

    #         # Add a main title to the entire figure
    #         fig.suptitle(
    #             f"{bs.id.upper()} \n Aggregated Kalman Filter and Queue Simulation Results",
    #             fontsize=20
    #         )

    #         # Adjust layout to prevent overlap
    #         # plt.tight_layout(rect=[0, 0.03, 1, 0.92]) #0.95
    #         plt.tight_layout(rect=[0, 0, 1, 0.97])#0.92
    #         #plt.tight_layout()

    #         # Display the aggregated figure
    #         plt.show()




        
    