from simulation_files.utils.helper_functions import *
from simulation_files.simulator_backbone.network_instances import *
from simulation_files.simulator_backbone.logger import *

class Controller:

    def __init__(self,bss_pos:np.ndarray,outer_window_p:np.ndarray,
                 inner_window_p :np.ndarray,bss_voronoi:Voronoi) -> None:
        """
        Initializes the Controller class with base station positions, boundary points, and other attributes.

        Params:
            bss_pos (np.ndarray): Array of base station positions.
            outer_window_p (np.ndarray): Array of boundary points for the grid.
        """
        self.bs_map = [BS(id=fr'BS_{i}',position=pos, region=None) for i, pos in enumerate(bss_pos)] 
        self.sos_map = []
        self.ues_map = []

        self.cmap = viridis(np.linspace(0, 1, len(self.bs_map)))  # Create the colormap
        self.outer_window_p = outer_window_p
        self.inner_window_p = inner_window_p
        self.bss_voronoi = bss_voronoi
        self.bss_pos = bss_pos
        self.verbose = None
        self.id_tracker = 0
        self.grid_polig = self.gen_grid_polygon()
    
    def get_key(self,dict, val):
        """
        Find the key in the dictionary that matches the given value.

        Params:
            dict (dict): Dictionary to search.
            val (np.ndarray): Value to search for in the dictionary.

        Return:
            str: The key associated with the value if found, or "key doesn't exist".
        """
        for key, value in dict.items():
            if (val == value).all():
                return key

        return "key doesn't exist"
    
    def get_grid_coord(self):
        """
        Get the minimum and maximum x and y coordinates from the boundary points.

        Params:
            None

        Return:
            tuple: A tuple containing (min_x, min_y, max_x, max_y) as the grid coordinates.
        """
        min_x = np.min(self.inner_window_p[:, 0]).astype(float)
        min_y = np.min(self.inner_window_p[:, 1]).astype(float)
        max_x = np.max(self.inner_window_p[:, 0]).astype(float)
        max_y = np.max(self.inner_window_p[:, 1]).astype(float)

        return min_x,min_y,max_x,max_y
    
    def sample_bss(self,n_bss_:int):
        min_x,min_y,max_x,max_y = self.get_grid_coord()
        x_d_ = max_x - min_x #width
        y_d_ = max_y - min_y #height
        return np.array([min_x, min_y] + rand(n_bss_, 2) * [x_d_, y_d_])

    def gen_grid_polygon(self):
        """
        Generate a rectangular polygon from the boundary points.

        Params:
            None

        Return:
            Polygon: A Shapely polygon object representing the grid.
        """
        min_x,min_y,max_x,max_y = self.get_grid_coord()
        coords = ((min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x,min_y), (min_x, min_y))
        return Polygon(coords)


    def check_bs_region(self,polygon, point):
        """
        Check whether a point is inside a given polygon (Voronoi cell).

        Params:
            polygon (np.ndarray): Vertices of the polygon.
            point (np.ndarray): The point to check.

        Return:
            bool: True if the point is inside the polygon, False otherwise.
        """
        return Polygon(polygon).contains(Point(point))
    
    def cap_voronoi_vertices(self):
        """
        Cap Voronoi vertices to stay within the grid boundary.

        """
        # Get grid coordinates
        min_x, min_y, max_x, max_y = self.get_grid_coord()
        
        # Cap the Voronoi vertices
        self.bss_voronoi.vertices = np.clip(self.bss_voronoi.vertices, 
                                   a_min=[min_x, min_y], 
                                   a_max=[max_x, max_y])
        
    def reinit_voronoi(self):
        n_bss_ = len(self.bs_map)
        self.bss_pos = self.sample_bss(n_bss_=n_bss_)
        extended_bss_pos = np.vstack((self.bss_pos, self.outer_window_p))
        self.bss_voronoi = Voronoi(extended_bss_pos)
        #New
        self.bs_map = [BS(id=fr'BS_{i}',position=pos, region=None) for i, pos in enumerate(self.bss_pos)] 
        #self.cap_voronoi_vertices()

    def cap_regions(self):
        for bs in self.bs_map:
            r = bs.region
            bs.region = Polygon(r).intersection(self.grid_polig)

    def init_net(self):
        """Initialize the network by assigning Voronoi regions to base stations."""
        # print(regions)
        while True:
            regions=[]
            for region in self.bss_voronoi.regions: #I might have more regions that BSs
                if len(region) > 0 and -1 not in region:
                    r = [self.bss_voronoi.vertices[idx] for idx in region]
                    regions.append(r) #this are already polygons
            matcher = {}
            reinit = False
            
            for bs in self.bs_map:
                # Check if base station position is inside any Voronoi region
                # for r in regions:
                #     print(r)
                #     print('---')
                bs_regions = [self.check_bs_region(polygon=r, point=bs.position) for r in  regions]

                if sum(bs_regions) != 1:  # If more than one or 0 region matches
                    self.reinit_voronoi()
                    reinit = True
                    break  # Reinitialize and restart the loop
                else:
                    # Assign region to the base station
                    #here is wrong
                    i = np.where(bs_regions)[0][0]
                    # print(i)
                    # print(bs.id)
                    matcher[bs.id] = regions[i]

            if not reinit:
                # Finalize assignment of regions to base stations
                for i in matcher.keys():  # Use keys() instead of iterkeys()
                    for bs in self.bs_map:
                        if bs.id == i:
                            bs.region = matcher[i]
                break  # Exit the loop after successful initialization

        
    def generate_particles_in_cell(self,polygon:np.ndarray, bs:BS, num_points:int,epsilon: float = 0.0):
        """
        Generate random points within the given polygon (Voronoi cell).

        Params:
            polygon (np.ndarray): Vertices of the polygon defining the Voronoi cell.
            num_points (int): Number of points to generate inside the polygon.
            bs_coord (np.ndarray): Coordinates of the base station related to this polygon.
            epsilon (float): Optional margin to avoid placing points near the boundary (default is 0.0).

        Return:
            np.ndarray: Array of generated points within the polygon.
        """
        #if you pass the BS object then you get access to his polygon
        
        # pp = Polygon(polygon).intersection(self.grid_polig)
        pp = Polygon(polygon)
        # print(pp)
        min_x, min_y, max_x, max_y = pp.bounds
        
        coords = []

        while len(coords) < num_points:
            point = uniform(low=[min_x, min_y], high=[max_x, max_y], size=(1, 2))
            if self.check_bs_region(polygon=polygon, point=point):
                d = pp.exterior.distance(Point(point))
                if  d > epsilon:
                    if self.verbose:
                        print('BS:',bs.id,'-',"Distance Item-BS's region:", round(d,3), 'm')
                    coords.append(point[0])

        return np.array(coords)
    
    def generate_items_voronoi(self,n_bss: int, obj_tag:str, lambda_: float = None, 
                               n_items: int = None, verbose: bool = False,epsilon: float = 0.0,):
        
        """
        Generate random items (particles) in Voronoi cells for each base station (BS).

        Params:
            n_bss (int): Number of base stations.
            lambda_ (float, optional): The intensity of items to generate per unit area. Defaults to None.
            n_items (int, optional): Total number of items to generate. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            epsilon (float, optional): The margin to avoid placing points near the polygon edges. Defaults to 0.0.

        Return:
            np.ndarray: An array of generated item positions within the Voronoi cells.
        """

        if lambda_ is None and n_items is None:
            raise ValueError('Provide at least the PPP intensity (lambda_) or the number of items (n_items) to generate in the Voronoi cell.')
        
        self.verbose = verbose

        # Determine the number of items to generate
        if n_items is not None:
            n_it_per_bs = np.ceil(n_items / n_bss).astype(int)  # Ensure it's an integer
        else:
            intensity = np.ceil(lambda_ / n_bss).astype(int)
            n_it_per_bs = [poisson(intensity) for _ in range(n_bss)]

        if self.verbose:
            print('n_it_per_bs:', n_it_per_bs, '\n')

        pos = []
        #self.cap_voronoi_vertices()  # Assuming this function is defined elsewhere
        self.init_net()
        self.cap_regions()
        for bs in self.bs_map:
            polygon = bs.region
            #polygon = [self.bss_voronoi.vertices[idx] for idx in region]
            to_gen = n_it_per_bs[bs.id] if isinstance(n_it_per_bs, list) else int(n_it_per_bs)
            items_in_cell = self.generate_particles_in_cell(polygon=polygon, bs=bs, num_points=to_gen, epsilon=epsilon)
            pos.append(items_in_cell)
            for p in items_in_cell:
                self.id_tracker +=1
                if obj_tag == 'so':
                    so = Sensed_Obj(id=f'SO_{self.id_tracker}', position=p, region=polygon, bs=bs)
                    self.sos_map.append(so)
                    bs.sos_.append(so)
                elif obj_tag == 'ue':
                    ue = User(id=f'UE_{self.id_tracker}', position=p, region=polygon, bs=bs)
                    self.ues_map.append(ue)
                    bs.ues_.append(ue)
                else:
                    raise ValueError('Undefined object type. Either "so" or "ue".')
        
        return np.vstack(pos) if pos else np.empty((0, 2))


    def plot_network(self, sos_pos: np.ndarray, ues_pos: np.ndarray, title:str):
        """
        Plot the network with base stations (BSs), sensed objects (SOs), and user equipment (UEs).

        Params:
            sos_pos (np.ndarray): Array of sensed object positions.
            ues_pos (np.ndarray): Array of user equipment positions.
            title (str): Title for the plot.

        Return:
            None
        """

        # for i, region in enumerate(self.bs_regions.values()):
        #     x_region, y_region = zip(*region)  # Unzip the region points into x and y coordinates
        #     plt.fill(x_region, y_region, color=self.cmap[i], alpha=0.5, label=f'Region {i}')
        

        # Annotate base stations with their IDs
        
        # for i,bs in enumerate(self.bs_map):
        #     bs_id = bs.id
        #     bs_coord = bs.position
        #     region = bs.region
            #plt.text(bs_coord[0] - 20, bs_coord[1] - 20, bs_id, fontsize=10, ha='right', va='center', color='black')

        for region in self.bss_voronoi.regions:
            if len(region) > 0 and -1 not in region:
                r = [self.bss_voronoi.vertices[idx] for idx in region]
                tmp = [self.check_bs_region(polygon=r, point=p) for p in self.bss_pos]
                if any(tmp):
                    i = np.where(tmp)[0][0]
                    x_region, y_region = zip(*r)  # Unzip the region points into x and y coordinates
                    plt.fill(x_region, y_region, color=self.cmap[i], alpha=0.5, )#label=f'Region {i}'

        # Plot base stations as blue triangles
        #self.bss_pos = np.array([bs.position for bs in self.bs_map])
        plt.scatter(self.bss_pos[:, 0], self.bss_pos[:, 1], c='blue', label='BSs', marker='^', s=100)

        # Plot sensed objects as red 'x'
        if sos_pos is not None:
            # Annotate base stations with their IDs
            # for so in self.so_map:
            #     so_coord = so.position
            #     so_id = so.id
                #plt.text(so_coord[0] - 20, so_coord[1] - 20, so_id, fontsize=10, ha='right', va='center', color='black')
            plt.scatter(sos_pos[:, 0], sos_pos[:, 1], c='red', label='SOs', marker='s')

        # Plot user as green 'o'
        if ues_pos is not None:
            # for ue in self.ues_map:
            #     ue_coord = ue.position
            #     ue_id = ue.id
                #plt.text(ue_coord[0] - 20, ue_coord[1] - 20, ue_id, fontsize=10, ha='right', va='center', color='black')
            plt.scatter(ues_pos[:, 0], ues_pos[:, 1], c='green', label='UEs', marker='h')

        # Add legend and title
        # if len(self.bss_pos) <= 5:
        plt.legend(loc='upper left')
        plt.title(title)

        min_x,min_y,max_x,max_y = self.get_grid_coord()

        plt.xlim(min_x,max_x)

        plt.ylim(min_y,max_y)

        # Show the plot
        # plt.show()

        # plt.close()
    
    #bss:list
    #self.bs_map = bss

    def plot_spatial_heatmap(self,save_dir:str, title:str, tail:bool = True):

        high_ = 'm'
        low_ = 'g'
        intermediate_ = 'y'

        #define color mapping based on correllation measure -> green (highcorrelation), yellow (small correlation), red (indipended)
        def assign_color(item):
            """
            Assigns a color based on the 2D coordinates (x, y).

            Parameters:
            - item (tuple or list): A tuple or list containing two numerical values (x, y).

            Returns:
            - str: The color corresponding to the quadrant.
            
            Raises:
            - ValueError: If any of the coordinates are outside the [0, 1] range.
            """
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("Item must be a tuple or list with exactly two numerical values (x, y).")
            
            x, y = item  # Unpack the two dimensions
            
            # Take absolute values
            x = abs(x)
            y = abs(y)
            
            # Validate the range
            if x > 1 or y > 1:
                raise ValueError("KF's covariance norm and Workload should be normalized within [0,1].")
            
            # Assign colors based on the quadrants
            if x <= 0.5 and y <= 0.5:
                return low_
            elif x > 0.5 and y > 0.5:
                return high_
            else:
                return intermediate_


        colr_map = {bs: assign_color(bs.region_mark) for bs in self.bs_map}

        # for bs, color in colr_map.items():
        #     x_region, y_region = zip(*bs.region)
        #     plt.fill(x_region, y_region, color=color, alpha=0.5)
        if len(self.bs_map) < 20:
            for bs in self.bs_map:
                bs_coord = bs.position
                plt.text(bs_coord[0] - 20, bs_coord[1] - 20, bs.id, fontsize=10, ha='right', va='center', color='black')

        # Plot each region with the assigned color
        for bs, color in colr_map.items():
            # Access the exterior coordinates of the Polygon
            if hasattr(bs.region, 'exterior') and hasattr(bs.region.exterior, 'coords'):
                x_region, y_region = zip(*bs.region.exterior.coords)
                plt.fill(x_region, y_region, color=color, alpha=0.5)
            else:
                raise TypeError(f"The region of BS {bs} is not a valid Polygon with exterior coordinates.")
            

        # # Determine inequality strings for W and Sigma
        # str_w = r'$W_\infty \geq \tau_W$' if w_above else r'$W_\infty \leq \tau_W$'
        # str_sigma = r'$\Sigma_\infty \geq \tau_\Sigma$' if kf_above else r'$\Sigma_\infty \leq \tau_\Sigma$'

        # # Construct the indicator string
        # indct_str = f'$1_{{{str_w},{str_sigma}}}$'

        # # Define region mappings
        # region_mapping = {
        #     'A': indct_str + r'$ = (1,1)$',
        #     'B': indct_str + r'$ \in \{(0,1),(1,0)\}$',
        #     'C': indct_str + r'$ = (0,0)$',
        # }
        # Determine inequality strings for W and Sigma
        # Determine inequality strings for W and Sigma
        str_w = r'K \geq \tau_W' if tail else r'K \leq \tau_W'
        str_sigma = r'U \geq \tau_\Sigma' if tail else r'U \leq \tau_\Sigma'

        # Construct the indicator string
        indct_str = rf'(1_{{{str_w}}},1_{{{str_sigma}}})'

        # Define region mappings
        region_mapping = {
            'Violet': rf'${indct_str} = (1,1)$',
            'Yellow': rf'${indct_str} \in \{{(0,1),(1,0)\}}$',
            'Green': rf'${indct_str} = (0,0)$',
        }



        
        #Region mapping table
        # str_w = None
        # str_sigma = None
        # if w_above:
        #     str_w = r'$W_\infty \geq \tau_W$'
        # else:
        #     str_w = r'$W_\infty \leq \tau_W$'

        # if kf_above:
        #     str_sigma = r'$\Sigma_\infty \geq \tau_\Sigma$'
        # else:
        #     str_sigma = r'$\Sigma_\infty \leq \tau_\Sigma$'

        # indct_str = rf'$(1_{str_w},1_{str_sigma})$'

        # region_mapping = {
        #     'A': indct_str + r'$ = (1,1)$',
        #     'B': indct_str + r'$ \in \{(0,1),(1,0)\}$',
        #     'C': indct_str + r'$ = (0,0)$',
        #     # 'B': r'$1_{(W_{\infty},\Sigma_{\infty}) \in [0,1] \times [1,0] \cup [1,0] \times [0,1]} + 1_{(W_{\infty},\Sigma_{\infty}) \in [1,0] \times [0,1]}$',
        #     # 'B': r'$1_{(W_{\infty},\Sigma_{\infty}) \in [0,1] \times [1,0] \cup [1,0] \times [0,1]}$',
        #     # 'C': r'$1_{(W_{\infty},\Sigma_{\infty}) \in [0,0]^2 }$'
        # }


        generate_log(hyperparams=region_mapping, title='Network heatmap reference',header=['Color', 'Region Description'], save_dir=save_dir)


        plt.scatter(self.bss_pos[:, 0], self.bss_pos[:, 1], c='blue', marker='^', s=100) #label='BSs'

        # Create custom legend patches for the color ranges
        high_load_patch = mpatches.Patch(color=high_, label='A')
        low_load_patch = mpatches.Patch(color=intermediate_, label='B')
        independent_patch = mpatches.Patch(color=low_, label='C')
        
        # Combine the custom patches with existing handles
        #plt.legend(handles=[high_load_patch, low_load_patch, independent_patch])# plt.Line2D([], []), prop={'size': 6} marker='^', color='blue', linestyle='None', markersize=10, label='Base Stations'

        # plt.legend()
        plt.title(title)

        min_x,min_y,max_x,max_y = self.get_grid_coord()

        # Adjust axis ticks to reflect the offset and show desired x_min, 0, x_max and y_min, 0, y_max
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        # Manually set the ticks for x and y axes
        xticks = [min_x, 0, max_x]
        yticks = [min_y, 0, max_y]

        # Set x and y ticks explicitly
        plt.xticks(xticks)
        plt.yticks(yticks)

        # Label the axes
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.savefig(f'{save_dir}/network_heatmap.png')
        #os.path.join(save_dir, "steady_state_heatmap.png")

        # Show the plot
        plt.show()

        plt.close()

        # for region in self.bss_voronoi.regions:
        #     if len(region) > 0 and -1 not in region:
        #         r = [self.bss_voronoi.vertices[idx] for idx in region]
        #         tmp = [self.check_bs_region(polygon=r, point=p) for p in self.bss_pos]
        #         if any(tmp):
        #             i = np.where(tmp)[0][0]
        #             x_region, y_region = zip(*r)  # Unzip the region points into x and y coordinates
        #             plt.fill(x_region, y_region, color=self.cmap[i], alpha=0.5, )#label=f'Region {i}'

        # # Plot base stations as blue triangles
        # #self.bss_pos = np.array([bs.position for bs in self.bs_map])
        
    
    def plot_trajectory(self, save_dir:str, sos_traj: dict, ues_traj: dict, title:str):
        """
        Plot the network with base stations (BSs), sensed objects (SOs), and user equipment (UEs).

        Params:
            sos_pos (np.ndarray): Array of sensed object positions.
            ues_pos (np.ndarray): Array of user equipment positions.
            title (str): Title for the plot.

        Return:
            None
        """
        
        #shift_array = lambda arr, offset: np.array([[x[0] + offset[0], x[1] + offset[1]] for x in arr])
    
        # Annotate base stations with their IDs
        # for i,bs in enumerate(self.bs_map):
        #     bs_id = bs.id
        #     bs_coord = bs.position
        #     region = bs.region
        #     plt.text(bs_coord[0] - 0.5, bs_coord[1], str(bs_id), fontsize=10, ha='right', va='center', color='black')

        #TO COLOR THE REGION
        # for region in self.bss_voronoi.regions:
        #     if len(region) > 0 and -1 not in region:
        #         r = [self.bss_voronoi.vertices[idx] for idx in region]
        #         tmp = [self.check_bs_region(polygon=r, point=p) for p in self.bss_pos]
        #         if any(tmp):
        #             i = np.where(tmp)[0][0]
        #             x_region, y_region = zip(*r)  # Unzip the region points into x and y coordinates
                    #plt.fill(x_region, y_region, color=self.cmap[i], alpha=0.5, label=f'Region {i}')

        # Plot base stations as blue triangles
        
        # self.bss_pos = shift_array(self.bss_pos, offset_xy)
        # print(bss_pos)

        #self.bss_pos = np.array([bs.position for bs in self.bs_map])
        plt.scatter(self.bss_pos[:, 0], self.bss_pos[:, 1], c='blue', label='BSs', marker='^', s=100)
        plt.scatter(0,0,marker='.', color='k', label='SO')
        plt.scatter(0,0,marker='o', color='white')
        plt.scatter(0,0,marker='x', color='k', label='UE')
        plt.scatter(0,0,marker='s', color='white')

        # Plot sensed objects as red 'x'
        if sos_traj is not None:
            so_ids = sos_traj.keys()
            n_sos = len(so_ids)
            cmap = plt.get_cmap('tab20b', n_sos)
            for i, (so_id, traj) in enumerate(sos_traj.items()):
                color = cmap(i / (n_sos - 1))
                #color = 'k'
                traj = np.array(traj)  # Convert to array if needed
                # traj = shift_array(traj, offset_xy)
                plt.scatter(traj[:, 0], traj[:, 1], c=color, marker='.') # label=f'{so_id}'
                plt.plot(traj[:, 0], traj[:, 1], c=color, linestyle=':', linewidth=1)

        # Plot user as green 'o'

        #If users move
        if ues_traj is not None:
            ues_ids = ues_traj.keys()
            n_ues = len(ues_ids)
            cmap = plt.get_cmap('RdBu', n_ues)
            for i, (ues_id, traj) in enumerate(ues_traj.items()):
                color = cmap(i / (n_ues - 1))
                #color = 'k'
                traj = np.array(traj)  # Convert to array if needed
                # traj = shift_array(traj, offset_xy)
                plt.scatter(traj[:, 0], traj[:, 1], c=color, marker='x') # label=f'{so_id}'
                plt.plot(traj[:, 0], traj[:, 1], c=color, linestyle=':', linewidth=1)

        #If users are still
        # if ues_traj is not None:
        #     # ues_traj = shift_array(ues_traj, offset_xy)
        #     plt.scatter(ues_traj[:, 0], ues_traj[:, 1], c='green', label='UEs', marker='o')

        # Add legend and title
        # if len(self.bss_pos) <= 5:
        #    plt.legend()
        plt.legend()
        plt.title(title)

        min_x,min_y,max_x,max_y = self.get_grid_coord()


        # Adjust axis ticks to reflect the offset and show desired x_min, 0, x_max and y_min, 0, y_max
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        # Manually set the ticks for x and y axes
        xticks = [min_x, 0, max_x]
        yticks = [min_y, 0, max_y]

        # Set x and y ticks explicitly
        plt.xticks(xticks)
        plt.yticks(yticks)

        # Label the axes
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.savefig(f'{save_dir}/network_trajectories.png')

        # Show the plot
        plt.show()

        plt.close()


