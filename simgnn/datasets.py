from os import path, listdir
from glob import glob

import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from simgnn.datautils import load_array, load_graph

dtype = torch.float32

class CellData(Data):
    '''Cell monolayer graph data object. Same as `Data` but with cells.'''
    def __init__(self, y_cell=None, num_cells=None,
                 node2cell_index=None, cell2node_index=None, **kwargs):
        super(CellData, self).__init__(**kwargs)
        self.node2cell_index = node2cell_index
        self.cell2node_index = cell2node_index
        self.y_cell = y_cell
        self.__num_cells__ = num_cells
    
    @property
    def num_cells(self):
        if self.__num_cells__!=None:
            return self.__num_cells__
        if self.node2cell_index!=None:
            print('Number of cells is inferred from `node2cell_index`!')
            return self.node2cell_index[1].max()+1
        if self.cell2node_index!=None:
            print('Number of cells is inferred from `cell2node_index`!')
            return self.cell2node_index[0].max()+1
    
    @num_cells.setter
    def num_cells(self,val):
        self.__num_cells__ = val
        
    def __inc__(self, key, value):
        if key == 'node2cell_index':
            return torch.tensor([[self.num_nodes], [self.num_cells]])
        if key == 'cell2node_index':
            return torch.tensor([[self.num_cells], [self.num_nodes]])
        else:
            return super(CellData, self).__inc__(key, value)
    
    def __cat_dim__(self, key, value):
        if key == 'node2cell_index' or key == 'cell2node_index':
            return -1
        else:
            return super(CellData, self).__cat_dim__(key, value)


class VertexDynamics(Dataset):
    def __init__(self, root, window_size=5, transform=None, pre_transform=None):
        '''
        Assumes `root` dir contains folder named `raw` with all vertex dynamics simulation results
        for tracing vertex trajectories, building graphs, and variables for computing edge tensions 
        and cell pressures.
        - Velocities are approximated as 1st order differences of positions `x` in subsequent frames:
          `velocity(T+0) = x(T+1) - x(T+0)`.
        - Use `pre_transform` for normalising and pre-processing dataset(s).
        
        Arg-s:
        - root : path to a root directory that contains folder with raw dataset(s) in a folder named "raw".
        Raw datasets should be placed into separate folders each containing outputs from a single simulation.
        E.g. root contains ["raw", "processed", ...], and in folder "raw/" we should have ["simul1", "simul2", ...]
        - window_size : number of past velocities to be used as node features 
        `[x(T+0)-x(T-1), x(T-1)-x(T-2),..., x(T-window_size+1)-x(T-window_size)]`, where `x(T)` is node position at time `T`.
        - transform :  transform(s) for graph datasets (e.g. from torch_geometric.transforms )
        - pre_transform : transform(s) for data pre-processing (resulting graphs are saved in "preprocessed" folder)
        and used as this dataset's sample graphs.
        '''
        self.raw_dir_path = path.join(root,'raw')
        assert path.isdir(self.raw_dir_path), f'Folder "{root}" does not contain folder named "raw".'
        
        self.window_size = window_size
        
        super(VertexDynamics, self).__init__(root, transform, pre_transform)
        # super's __init__ runs process() [and download() if defined].

    @property
    def raw_file_names(self):
        raw_dirs = [folder_i for folder_i in listdir(self.raw_dir_path) 
                    if path.isdir( path.join( self.raw_dir_path, folder_i))]
        #file_names = [path.join(dir_i, file_i) for dir_i in raw_dirs
        #              for file_i in listdir(path.join(self.raw_dir_path,dir_i))] 
        return raw_dirs

    @property
    def processed_file_names(self):
        '''
        Return list of pytorch-geometric data files in `root/processed` folder (`self.processed_dir`).
        '''
        # "last_idx" : last index of window in "vertex velocity" (for features)
        # last_idx=T-(2+window_size) --> num of processed frames: num_of_frames=last_idx+1 
        nums_of_frames = [ (path.basename(raw_path),
                            load_array(path.join(raw_path,'simul_t.npy')).shape[0]-(2+self.window_size)+1 
                           ) for raw_path in self.raw_paths]
        file_names = ['data_{}_{}.pt'.format(raw_path, t)
                      for raw_path, tmax in nums_of_frames for t in range(tmax)]
        return file_names

    def process(self):
        '''
        Assumptions:
        - the parent class init runs _process() and initialises all the required dir-s.
        - cell graph topology and number of nodes is constant w.r.t. to frames.
        '''
        for raw_path in self.raw_paths:
            # simulation instance in "raw_path"
            
            # T+0 Cell pressures (torch.Tensor)
            cell_presrs = self.cell_pressures(area_path = path.join(raw_path, 'simul_Area.npy'),
                                              a0_path = path.join(raw_path, 'simul_A0.npy'),
                                              ka_path = path.join(raw_path, 'simul_Ka.npy'))
            
            # Load node positions from raw_path and convert to node attrib-s and targets.
            node_pos, X_node, Y_node = self.pos2nodeXY(pos_path = path.join(raw_path,'simul_vtxpos.npy') )
            
            # monolayer graph (topology)
            mg_dict = load_graph(path.join(raw_path,'graph_dict.pkl'))
            
            # edge indices 
            edges = torch.tensor(mg_dict['edges'],dtype=torch.long) # assume constant w.r.t "t"
            edge_index = torch.cat( [edges.T.contiguous(), edges.fliplr().T.contiguous()], axis=1)
            
            # cell-to-node and node-to-cell "edge indices"
            cell2node_index = self.cell2edge(edges=edges, cells=mg_dict["cells"]) # cell_id-node_id pairs
            node2cell_index = cell2node_index[[1,0]].contiguous() # node_id-cell_id pairs
            
            sim_name = path.basename(raw_path) # folder name for the files
            N_nodes = node_pos.size(1) # assume constant w.r.t. "t"
            N_cells = max(mg_dict['cells'].keys())+1 # num_of_cells assume constant w.r.t. "t"
            
            for t in range(node_pos.size(0)):
                data = CellData(num_nodes = N_nodes,
                                num_cells = N_cells,
                                edge_index = edge_index,
                                node2cell_index = node2cell_index, cell2node_index = cell2node_index,
                                pos = node_pos[t], x = X_node[t], y = Y_node[t],
                                cell_pressures = cell_presrs[t]
                               )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, path.join(self.processed_dir, 'data_{}_{}.pt'.format(sim_name, t)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load( path.join( self.processed_dir, self.processed_file_names[idx]))
        return data
    
    def pos2nodeXY(self, pos_path=None):
        '''
        Load 'simul_vtxpos.npy' and pre-process it into windowed data of velocities (1st differences) using `self.window_size` number of past velocities (frames) as node features.
         
        - pos_path: path to vertex positions file, e.g. 'simul_vtxpos.npy', containing an array with shape `(frames)xNx2`.
        
        Returns: {dtype : `torch.float32`}
        - node_pos: node positions : shape (num_of_frames)xNx2
        - X_node: node velocities ordered from `T-window_size` to `T-1` : shape (num_of_frames)xNx(window_size)x2
        - Y_node : `T+0` vertex velocities : shape (num_of_frames)xNx2
        
        Where the `num_of_frames=last_idx+1`, and `last_idx=frames-(2+window_size)` is the last index of window in vx_vel (full array of 1st differences of the node positions with "frames-1" number of frames, i.e. vx_pos.shape[0]==frames).
        '''
        # vertex trajectories:(Frames,Vertices,Dims)=TxNx2
        vx_pos = load_array(pos_path) #TxNx2
        vx_vel = np.diff(vx_pos,n=1,axis=0) # velocity(1st diff approx):(T-1)xNx2

        # T+0 vertex positions
        node_pos = torch.from_numpy( vx_pos[self.window_size:-1]).type(dtype) # (num_of_frames)xNx2

        # (T-window_size) to (T-1) velocities : (num_of_frames)xNx(window_size)x2
        X_node = torch.from_numpy(np.stack([ vx_vel[k:k+self.window_size].transpose((1,0,2))
                                            for k in range(vx_pos.shape[0]-(2+self.window_size)+1)])
                                  ).type(dtype)
        # T+0 vertex velocities
        Y_node = torch.from_numpy(vx_vel[self.window_size:]).type(dtype) # (num_of_frames)xNx2
        return node_pos, X_node, Y_node
    
    def cell2edge(self, edges=None, cells=None):
        '''
        - edges: source-target vertex index pairs of directed edges (i.e. single copy of each edge in a graph)
        - cells: cells dict, cell indices (0...N_cells-1):int as keys and shifted edge indices (1..N_edges):int as values.
        Edges in cells must be ordered in order of their connection, indexing starts from 1 with negative indices indicating reversed order of vertices e.g. e[ID]=(v1,v2) and e[-ID]=(v2,v1). Edge indices in cells:`cells[ci]= [ID,...]` are converted to node (vertex) indices in "edges" tensor with `edge_ID=np.abs(ID)-1` ==> `(v1,v2) = edges[edge_ID]`, and order of the vertices is inferred from sign of `np.sign(ID)`: if 1 =>(v1,v2), elif -1 =>(v2,v1).
        '''
        return torch.cat([torch.stack( [torch.empty( len(cells[ci]), dtype=edges.dtype).fill_(ci),
                                       edges[np.abs(cells[ci])-1,np.sign(np.sign(cells[ci])-1)] ], dim=0)
                          for ci in cells ], dim=-1)
    
    def cell_pressures(self, area_path=None,a0_path=None,ka_path=None):
        '''
        Load numpy arrays from cell `Area`, target/equilibrium area `A0`, and "spring" constant `Ka` "*.npy" files, and pre-process these var-s as windowed data, and then computes cell pressures using these arrays. `press_c = -2*Ka*(A-A0)` for each cell.
        - area_path : cell areas w/ shape Frames x Cells
        - a0_path : equilibrium areas w/ shape Frames x Cells or (Frames,)
        - ka_path : area "spring constants" w/ shape Frames x Cells or (Frames,)
        
        Rerturns: {torch.Tensor, dtype:`torch.float32`}
        - cell_presrs : cell pressures w/ shape (num_of_frames)xCells, where the `num_of_frames=last_idx+1`, and `last_idx=Frames-(2+window_size)` is the last index of window in `vx_vel` (full array of 1st differences of the node positions with `Frames-1` number of frames, i.e. `vx_pos.shape[0]==Frames`).
        '''
        Area = load_array(area_path)[self.window_size:-1]
        A0 = load_array(a0_path)[self.window_size:-1]
        Ka = load_array(ka_path)[self.window_size:-1]
        
        cell_presrs = -2.0*Ka.reshape(Area.shape[0],-1)*(Area.reshape(Area.shape[0],-1) - A0.reshape(Area.shape[0],-1))
        return torch.from_numpy(cell_presrs).type(dtype)
