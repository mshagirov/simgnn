import numpy as np
import torch
from os import path
import datetime

from vertex_simulation.primitives import unit_hexagons, VoronoiRegions2Edges, Vertex, Monolayer
from vertex_simulation.simulation import Simulation_Honda, Simulation_Honda_t
from simgnn.datautils import *

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
dtype = torch.float32
print(f'Defaults:\n |-device: {device}\n |-dtype : {dtype}')


__SIM_NAME__= 'unimodal_wbasetens_sims'
mknewdir(path.join('./simgnn_data',__SIM_NAME__))

SIM_SAVE_DIR = path.join('./simgnn_data',__SIM_NAME__,'raw')
mknewdir(SIM_SAVE_DIR)
print('Outputs will be saved in \n',SIM_SAVE_DIR)

N_sim_runs = 7

seed_seq = np.random.SeedSequence(42+3)
rngs = [np.random.Generator( np.random.PCG64(s)) for s in seed_seq.spawn(N_sim_runs)] # RNGs

# Simulation  Prop-s
T=6100 # total number of iter-s
delta_T=0.00164 # time step 
sample_freq=100 # movie frame_rate = sample_freq x delta_T

# Edge oscillations
#   unit_of_time = 1 min = 1/Dt ; Dt=simulation time step size
N_peaks_max = 0.6 # maximum number of peaks per unit of time


for sim_k in range(N_sim_runs):
    torch.manual_seed(42+3 + sim_k);
    
    sim_name = datetime.datetime.now().strftime('%d%m%Y')+f'_sim{sim_k:03}'
    tissue_shape = rngs[sim_k].choice([8,10],(2,))

    v_x,regions = unit_hexagons(tissue_shape[0], tissue_shape[1]) # unit hexagons
    v_x += rngs[0].standard_normal(size=(v_x.shape[0], v_x.shape[1]))*.2 #perturb vertices
    
    edge_list,cells = VoronoiRegions2Edges(regions) #convert Voronoi regions to cells & edges
    # Define cell monolayer
    m = Monolayer(vertices = Vertex( v_x.copy().tolist(), dtype = dtype),
                  edges = torch.tensor(edge_list),
                  cells = cells)

    # # # Cell param-s # # #
    Ka_cells = torch.normal(mean=1.0,std=0.1,size=(len(m.cells),) ).clamp_(0.75,1.25).type(dtype).to(device)
    A0_cells = torch.normal(mean=2.3,std=0.25,size=(len(m.cells),) ).clamp_(1.6,3).type(dtype).to(device)
    P0_cells = torch.normal(mean=0,std=0.15,size=(len(m.cells),1) ).clamp_(0,).type(dtype).to(device)
    Kp_cells = torch.normal(mean=0.003,std=0.001,size=(len(m.cells),1) ).clamp_(0.0001,).type(dtype).to(device)

    # "lambda_ij" amplitudes ~ Normal(mu=2.1,s.d.=.5) [clamped]
    # oscillation freq-s; unit_of_time = 1 min = 1/Dt : Dt:simulation time step size
    lmd_ij = torch.normal(mean=1.0,std=1.0,size=( m.edges.size(0),1) ).clamp_(0,6).type(dtype).to(device)
    base_lmd_ij = 0.4+1.2*torch.rand((m.edges.size(0),1)).to(device)
    omega_ij = N_peaks_max*np.pi*torch.rand_like(lmd_ij) # in [0, N_peaks_max*pi]
    phase_ij = 0.5*np.pi*torch.rand_like(lmd_ij) # in [0, 0.5*pi]; Phase --> cos2 value at t=0
    
    sim_params = {'Ka': lambda mg,t: Ka_cells, 'A0': lambda mg,t: A0_cells,
                  'Kp': lambda mg,t: Kp_cells, 'P0': lambda mg,t: P0_cells,
                  'Lambda_ij': lambda mg,t: base_lmd_ij + lmd_ij*torch.cos(omega_ij*t+phase_ij)**2 }

    m.vertices.requires_grad_(True) # grad tracking
    m.to_(device) # to cuda or cpu
    sim = Simulation_Honda_t(m = m, params = sim_params)
    
    print(f'\n{sim_name} {tissue_shape[0]}x{tissue_shape[1]}')
    t, verts_t, Energies_maxSpeeds = sim.sample_trajectory(T=T, delta_T=delta_T, sample_freq=sample_freq, print_freq=sample_freq)
    
    # convert torch.Tensors to [Time x Vertices x 2] numpy array
    # verts_t = np.array([vt.numpy() for vt in verts_t])
    
    dataDir = path.join(SIM_SAVE_DIR,sim_name)
    mknewdir(dataDir);
    print(f'Saving in {dataDir}')
    
    # save simulation results
    write_array(path.join(dataDir,'t_Energy_maxSpeed.npy'), np.array(Energies_maxSpeeds))
    write_array(path.join(dataDir,'simul_t.npy'), np.array(t))
    write_array(path.join(dataDir,'simul_vtxpos.npy'), np.array([vt.numpy() for vt in verts_t]))

    # save monolayer's graph and cells
    write_graph(path.join(dataDir,'graph_dict.pkl'), m)

    # disable grad tracking
    m.vertices.requires_grad_(False) # grad tracking
    # move to cpu, assuming Simul-n param-s don't depend on vertices
    m.to_(torch.device('cpu')) # to cuda or cpu

    # cell shape parameters: area, perimeter, edge lengths
    cell_params = {k:[] for k in ['Area', 'Perimeter', 'Length']} 
    sim_param_vals = {k:[] for k in sim_params.keys()}
    # (re-)compute areas and perimeters for sampled frames
    for ti,vx in zip(t, verts_t):
        m.vertices.x = vx.cpu() # set new vertex positions
        # cell params for vx, t:
        cell_params['Area'].append(m.area().cpu().numpy())
        cell_params['Perimeter'].append(m.perimeter().cpu().numpy())
        cell_params['Length'].append(m.length().cpu().numpy())
        # simulation params for vx,t (usually depend only on t)
        for k in sim_param_vals:
            # here sim_params[k] doesn't depend on `m`, otherwise compute on "device"
            k_val = sim_params[k](m, ti)
            if isinstance(k_val, torch.Tensor):
                k_val = k_val.cpu().numpy()
            sim_param_vals[k].append(k_val)

    # maybe can move some of the operations to the "Simulation" class' trajectory sampling
    # convert lists of arrays to nd-arrays
    cell_params = {k:np.array(cell_params[k]) for k in cell_params}
    sim_param_vals = {k:np.array(sim_param_vals[k]) for k in sim_param_vals}

    print('Please, re-check and verify that arrays in `sim_param_vals` are saved correctly')

    for p in [cell_params, sim_param_vals]:
        # p: parameters (either cell or simulation)
        for k in p:
            # k: name of param
            write_array(path.join(dataDir,f'simul_{k}.npy'),p[k])

print('--- Done ---')
