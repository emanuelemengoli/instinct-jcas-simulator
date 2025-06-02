from simulation_files.network_generator.radio_env import * 
from simulation_files.network_generator.voronoi_controller import *

# d_metric = 'm' #meters
# cov_side = lambda area, n_bss: np.sqrt(area/n_bss)
# cov_side_m = lambda area_km, n_bss: cov_side(area=area_km,n_bss=n_bss)*1000
# cov_area = lambda area, n_bss: (area/n_bss)
# n_bss_from_cov_side = lambda area, cov_side: (area/(cov_side**2))
# area_from_cov_side = lambda n_bss, cov_side: (cov_side**2)*n_bss
# km2_to_m2 = lambda km2: km2 * 10**6
# m2_to_km2 = lambda m2: m2 /10**6


# def get_new_area(half_base_m_old, half_height_m_old, lambda_bss_old, lambda_bss_new):

#     area_old_km2 = m2_to_km2(half_base_m_old*half_height_m_old*4)
#     area_new_km2 = m2_to_km2(area_from_cov_side(n_bss = lambda_bss_new, cov_side = cov_side(area=area_old_km2, n_bss=lambda_bss_old)*1000))

#     return area_new_km2

def build_tasselation(
        intensity_bss: int,
        intensity_ues: int = None,
        intensity_sos: int = None):
    
    x_max = Wx
    x_min = -Wx
    y_max = Hy
    y_min = -Hy

    x_d = x_max - x_min #width
    y_d = y_max - y_min #height
    area = x_d * y_d #total area m^2

    #Outer polygon (Omega)
    x_min_h = -Wx*20
    x_max_h = Wx*20
    y_min_h = x_min_h
    y_max_h = x_max_h


    n_bss = poisson(intensity_bss * area)
    n_ues = n_bss if intensity_ues is None else poisson(intensity_ues * area)
    n_sos = n_bss if intensity_sos is None else poisson(intensity_sos * area)

    bss_pos = [x_min, y_min] + rand(n_bss, 2) * [x_d, y_d]
    bss_pos = np.array(bss_pos)

    o_boundary_p = np.array([ 
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max]
        ])

    omega_boundary_p = np.array([ 
            [x_min_h, y_min_h],
            [x_min_h, y_max_h],
            [x_max_h, y_min_h],
            [x_max_h, y_max_h],
        ])

    # Combine base station positions with boundary points
    extended_bss_pos = np.vstack((bss_pos, omega_boundary_p))

    #extended_bss_pos = np.vstack((extended_bss_pos, o_boundary_p))
    # Perform Voronoi tessellation
    bss_voronoi = Voronoi(extended_bss_pos)

    cnt = Controller(bss_pos = bss_pos,outer_window_p=omega_boundary_p,inner_window_p =o_boundary_p,bss_voronoi=bss_voronoi)
    ues_pos = cnt.generate_items_voronoi(n_bss=n_bss,lambda_=None, n_items=n_ues, verbose=False,epsilon = 0, obj_tag='ue')
    sos_pos = cnt.generate_items_voronoi(n_bss=n_bss,lambda_=None, n_items=n_sos, verbose=False,epsilon = 0,obj_tag='so')

    return cnt, ues_pos, sos_pos


