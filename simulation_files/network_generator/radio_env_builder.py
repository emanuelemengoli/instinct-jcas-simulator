import os
import math

def generate_radio_env_file(
    FC: float, 
    W: float, 
    N: float, 
    ALPHA: float, 
    PW_TX: float, 
    G_TX_main: float, 
    G_TX_side: float, 
    G_RX: float, 
    net_width: float, 
    net_height: float
) -> None:
    """
    Generate the radio_env.py file with the provided parameters.
    Parameters:
    - FC: float - Carrier frequency in Hz
    - W: float - Channel bandwidth in Hz
    - N: int - Background noise in dBm/Hz
    - ALPHA: float - Path loss exponent (linear scale)
    - PW_TX: float - BS power in dBm
    - G_TX: float - TX Antenna gain in dBi
    - G_RX: float - RX Antenna gain in dBi
    - net_width: int - Width of the area (in meters)
    - net_height: int - Height of the area (in meters)
    """
    C = 3e8  # Speed of light in m/s
    # Define the content of the radio_env.py file with parameters
    content = f"""
# Radio Params:
C = {3e8} # Speed of light in m/s
F_C = {FC} #Hz Carrier frequency
W = {W} #Hz Channel Bandwidth
N = {N} #dBm/Hz background noise
G_0 = {(C/(4*math.pi*FC))**2}
#L0 = {20* math.log10((4*math.pi*FC)/C)} #- G_TX - G_RX #dBm/1m
ALPHA = {ALPHA} #lin-scale #urban scenario prev. 3.5
PW_TX = {PW_TX} #dBm
G_TX_MAIN = {G_TX_main} #dBi
G_TX_SIDE = {G_TX_side} #dBi
G_RX = {G_RX} #dBi
Wx = {net_width}
Hy = {net_height}
     """

    # Define the path for the new file
    file_path = 'simulation_files/network_generator/radio_env.py'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the content to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"File successfully created at: {file_path}")


