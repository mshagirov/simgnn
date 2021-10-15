from os import path, listdir

import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from simgnn.datautils import simple_moving_average, load_array, load_graph

dtype = torch.float32


def persistence_loss(graph_dataset):
    '''
    Computes a simple error for a prediction x_pos[T+0] = x_pos[T+1], i.e. velocity==0.

    Each elem-t in `graph_dataset` (an iterable) must contain T+0 node velocities in `graph_dataset[i].y`.

    velocity : dx[T+0] = x_pos[T+Lag] - x_pos[T+0] (generally Lag==1)
    persistence : loss(x_pos[T+0], x_pos[T+Lag]) = f(x_pos[T+Lag] - x[T+0]) <=> loss=f(dx[T+0]) | loss
                  in {MAE, MSE, RMSE}

    Returns:
        Dictionary {'mae':persistence_mae, 'mse':persistence_mse}
    '''
    dx_T0 = torch.cat([data.y for data in graph_dataset], axis=0)
    persistence_mae = np.mean(np.abs(dx_T0.numpy()))
    persistence_mse = np.mean(dx_T0.numpy()**2)
    return {'mae': persistence_mae, 'mse': persistence_mse}


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
        if self.__num_cells__ is not None:
            return self.__num_cells__
        if self.node2cell_index is not None:
            print('Number of cells is inferred from `node2cell_index`!')
            return self.node2cell_index[1].max()+1
        if self.cell2node_index is not None:
            print('Number of cells is inferred from `cell2node_index`!')
            return self.cell2node_index[0].max()+1

    @num_cells.setter
    def num_cells(self, val):
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
    '''For processing and working with vertex dynamics simulation output files.'''

    def __init__(self, root, window_size=5, transform=None, pre_transform=None, pre_filter=None,
                 smoothing=False, sma_lag_time=None):
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
        - window_size : number of past velocities to be used as node features ordered from `T-window_size` to `T-1`
                       `[x(T-window_size+1)-x(T-window_size),..., x(T-1)-x(T-2), x(T+0)-x(T-1)]`, where `x(T)` is
                       node position at time `T`. For more details see method `VertexDynamics.pos2nodeXY()`.
        - transform :  transform(s) for graph datasets (e.g. from torch_geometric.transforms ), used in parent class'
        loading method.
        - pre_transform : transform(s) for data pre-processing (resulting graphs are saved in "preprocessed" folder)
        and used as this dataset's sample graphs.
        - smoothing: If `True`, apply simple moving average on vertex positions. Computes `mean(x[T-sma_lag_time:T+1])`
                     along time dimension (must be axis=0 in vertex positions array).
        - sma_lag_time: a smoothing parameter, number of *past* vertex positions to use together with a current one in
                        computing a current average position (expected vertex position). Use this to denoise the vertex
                        trajectories. The current expected vertex position is approximated as
                        `mean({x[T-sma_lag_time], ..., x[T-1], x[T]})` (with available values).
        '''
        self.raw_dir_path = path.join(root, 'raw')
        assert path.isdir(self.raw_dir_path), f'Folder "{root}" does not contain folder named "raw".'

        self.window_size = window_size
        self.smoothing = smoothing
        self.sma_lag_time = sma_lag_time

        super(VertexDynamics, self).__init__(root, transform, pre_transform, pre_filter)
        # super's __init__ runs process() [and download() if defined].

    def apply_smoothing(self, v_pos):
        '''
        Apply smoothing along 0th axis of `v_pos` w/given `smoothing` and `sma_lag_time` parameters.
        '''
        return simple_moving_average(v_pos, self.sma_lag_time)

    @property
    def raw_file_names(self):
        raw_dirs = [folder_i for folder_i in listdir(self.raw_dir_path)
                    if path.isdir(path.join(self.raw_dir_path, folder_i))]
        # file_names = [path.join(dir_i, file_i) for dir_i in raw_dirs
        #              for file_i in listdir(path.join(self.raw_dir_path,dir_i))]
        return raw_dirs

    @property
    def processed_file_names(self):
        '''
        Return list of pytorch-geometric data files in `root/processed` folder (`self.processed_dir`).
        '''
        # "last_idx" : last index of window in "vertex velocity" (for features)
        # last_idx=T-(2+window_size) --> num of processed frames: num_of_frames=last_idx+1
        nums_of_frames = [(path.basename(raw_path),
                          load_array(path.join(raw_path, 'simul_t.npy')).shape[0] - (2+self.window_size) + 1
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

            # monolayer graph (topology)
            mg_dict = load_graph(path.join(raw_path, 'graph_dict.pkl'))

            # Load node positions from raw_path and convert to (windowed) node attrib-s and targets.
            node_pos, X_node, Y_node = self.pos2nodeXY(pos_path=path.join(raw_path, 'simul_vtxpos.npy'))

            # T+0 Cell pressures
            cell_presrs = self.cell_pressures(area_path=path.join(raw_path, 'simul_Area.npy'),
                                              a0_path=path.join(raw_path, 'simul_A0.npy'),
                                              ka_path=path.join(raw_path, 'simul_Ka.npy'))

            # T+0 Edge tensions
            edge_tensns = self.edge_tensions(mg_dict=mg_dict,
                                             lambdaij_path=path.join(raw_path, 'simul_Lambda_ij.npy'),
                                             perim_path=path.join(raw_path, 'simul_Perimeter.npy'),
                                             p0_path=path.join(raw_path, 'simul_P0.npy'),
                                             kp_path=path.join(raw_path, 'simul_Kp.npy'))

            # edge indices
            edges = torch.tensor(mg_dict['edges'], dtype=torch.long)  # assume constant w.r.t "t"
            edge_index = edges.T.contiguous()

            # cell-to-node and node-to-cell "edge indices"
            cell2node_index = self.cell2edge(edges=edges, cells=mg_dict["cells"])  # cell_id-node_id pairs
            node2cell_index = cell2node_index[[1, 0]].contiguous()  # node_id-cell_id pairs

            sim_name = path.basename(raw_path)  # folder name for the files
            N_nodes = node_pos.size(1)  # assume constant w.r.t. "t"
            N_cells = max(mg_dict['cells'].keys())+1  # num_of_cells assume constant w.r.t. "t"

            for t in range(node_pos.size(0)):
                data = CellData(num_nodes=N_nodes,
                                num_cells=N_cells,
                                edge_index=edge_index,
                                node2cell_index=node2cell_index, cell2node_index=cell2node_index,
                                pos=node_pos[t], x=X_node[t], y=Y_node[t],
                                cell_pressures=cell_presrs[t],
                                edge_tensions=edge_tensns[t]
                                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, path.join(self.processed_dir, 'data_{}_{}.pt'.format(sim_name, t)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

    def pos2nodeXY(self, pos_path=None):
        '''
        Load 'simul_vtxpos.npy' and pre-process it into windowed data of velocities (1st differences) using
        `self.window_size` number of past velocities (frames) as node features.

        - pos_path: path to vertex positions file, e.g. 'simul_vtxpos.npy', containing an array with
                    shape `(frames)xNx2`.

        Returns: {dtype : `torch.float32`}
        - node_pos: node positions : shape (num_of_frames)xNx2
        - X_node: node velocities ordered from `T-window_size` to `T-1` : shape (num_of_frames)xNx(window_size)x2
        - Y_node : `T+0` vertex velocities : shape (num_of_frames)xNx2

        Where the `num_of_frames=last_idx+1`, and `last_idx=frames-(2+window_size)` is the last index of window
        in vx_vel (full array of 1st differences of the node positions with "frames-1" number of frames,
        i.e. vx_pos.shape[0]==frames).
        '''
        # vertex trajectories:(Frames,Vertices,Dims)=TxNx2
        vx_pos = load_array(pos_path)  # TxNx2

        if self.smoothing:
            vx_pos = self.apply_smoothing(vx_pos)

        vx_vel = np.diff(vx_pos, n=1, axis=0)  # velocity(1st diff approx):(T-1)xNx2

        # T+0 vertex positions
        node_pos = torch.from_numpy(vx_pos[self.window_size:-1]).type(dtype)  # (num_of_frames)xNx2

        # (T-window_size) to (T-1) velocities : (num_of_frames)xNx(window_size)x2
        X_node = torch.from_numpy(np.stack([vx_vel[k:k+self.window_size].transpose((1, 0, 2))
                                            for k in range(vx_pos.shape[0] - (2 + self.window_size) + 1)])
                                  ).type(dtype)
        # T+0 vertex velocities
        Y_node = torch.from_numpy(vx_vel[self.window_size:]).type(dtype)  # (num_of_frames)xNx2
        return node_pos, X_node, Y_node

    def cell2edge(self, edges=None, cells=None):
        '''
        - edges: source-target vertex index pairs of directed edges (i.e. single copy of each edge in a graph)
        - cells: cells dict, cell indices (0...N_cells-1):int as keys and shifted edge indices (1..N_edges):int
        as values. Edges in cells must be ordered in order of their connection, indexing starts from 1 with
        negative indices indicating reversed order of vertices e.g. e[ID]=(v1,v2) and e[-ID]=(v2,v1). Edge
        indices in cells:`cells[ci]= [ID,...]` are converted to node (vertex) indices in "edges" tensor with
        `edge_ID=np.abs(ID)-1` ==> `(v1,v2) = edges[edge_ID]`, and order of the vertices is inferred from sign
        of `np.sign(ID)`: if 1 =>(v1,v2), elif -1 =>(v2,v1).
        '''
        return torch.cat([torch.stack([torch.empty(len(cells[ci]), dtype=edges.dtype).fill_(ci),
                                       edges[np.abs(cells[ci])-1, np.sign(np.sign(cells[ci])-1)]], dim=0)
                          for ci in cells], dim=-1)

    def cell_pressures(self, area_path=None, a0_path=None, ka_path=None):
        '''
        Load numpy arrays from cell `Area`, target/equilibrium area `A0`, and "spring" constant `Ka` "*.npy" files, and
        pre-process these var-s as windowed data, and then computes cell pressures using these arrays.
        `press_c = -2*Ka*(A-A0)` for each cell.

        - area_path : cell areas w/ shape Frames x Cells
        - a0_path : equilibrium areas w/ shape Frames x Cells or (Frames,)
        - ka_path : area "spring constants" w/ shape Frames x Cells or (Frames,)

        Rerturns: {torch.Tensor, dtype:`torch.float32`}
        - cell_presrs : cell pressures w/ shape (num_of_frames)xCells, where the `num_of_frames=last_idx+1`, and
        `last_idx=Frames-(2+window_size)` is the last index of window in `vx_vel` (full array of 1st differences
        of the node positions with `Frames-1` number of frames, i.e. `vx_pos.shape[0]==Frames`).
        '''
        Area = load_array(area_path)[self.window_size:-1]
        A0 = load_array(a0_path)[self.window_size:-1]
        Ka = load_array(ka_path)[self.window_size:-1]

        cell_presrs = -2.0*Ka.reshape(Area.shape[0], -1)*(
                            Area.reshape(Area.shape[0], -1) - A0.reshape(Area.shape[0], -1))
        return torch.from_numpy(cell_presrs).type(dtype)

    def edge_tensions(self, mg_dict=None, lambdaij_path=None, perim_path=None, p0_path=None, kp_path=None):
        '''
        Compute edge tensions (`tension_ij = dEnergy / dlength_ij`) :
        `tension_ij = Lambda_ij + 2*Kp_i*(perim_i - p0_i) + 2*Kp_j*(perim_j - p0_j)`

        Arg-s:
        - mg_dict : monolayer graph dict will cells and edges (from `graph_dict.pkl` file).
        - lambdaij_path : location of `Lambda_ij` "*.npy" file that contains an numpy array w/ shape
        Frames x Edges x 1, or Frames x Edges.
        - perim_path : location of a file with cell `Perimeters` numpy array w/
                       shape Frames x Cells x 1, or Frames x Cells.
        - p0_path : location of cell equilibrium perimeters `P0` array w/
                    shape (Frames,) or same shape as `Perimeters` array.
        - kp_path : location of cell perimeter "spring" constants, an array w/
                    shape (Frames,) or same shape as `Perimeters` array.

        Returns : {torch.Tensor, default dtype:`torch.float32`}
        - tensions : edge tensions w/ shape (num_of_frames)xEdges (how to compute "num_of_frames":
        see docs for `VertexDynamics.cell_pressures()`)
        '''
        # dict of cells sharing edges (neighbours)
        edge_cells = {ei: [] for ei in range(len(mg_dict['edges']))}
        for ci in mg_dict['cells']:
            for ei in np.abs(mg_dict['cells'][ci]) - 1:
                edge_cells[ei].append(ci)

        # Compute membrane tension (depends only on cell perimeter var-s)
        #
        Perims = load_array(perim_path)[self.window_size:-1]
        P0 = load_array(p0_path)[self.window_size:-1]
        Kp = load_array(kp_path)[self.window_size:-1]

        # compute cell-wise membrane tensions
        membrn_cells = 2.0*Kp.reshape(Perims.shape[0], -1)*(
                    Perims.reshape(Perims.shape[0], -1) - P0.reshape(Perims.shape[0], -1))

        # edge-wise sum of membrane tensions
        membrn_edge = np.concatenate([membrn_cells[:, edge_cells[ei]].sum(axis=1, keepdims=True)
                                      for ei in edge_cells], axis=1)

        # active tension due to (edge) contractility
        Lambda_ij = load_array(lambdaij_path)[self.window_size:-1].reshape(Perims.shape[0], -1)

        # total edge tensions
        edge_tensions = Lambda_ij + membrn_edge
        return torch.from_numpy(edge_tensions).type(dtype)


class HaraMovies(VertexDynamics):
    '''
    For working with processed Y. Hara et al. amnioserosa movies. Hara movies dataset does not have edge tensions
    and cell pressures, otherwise it is similar to the VertexDynamics dataset.
    '''

    def __init__(self, root, window_size=5, transform=None, pre_transform=None, pre_filter=None,
                 smoothing=False, sma_lag_time=None):
        '''
        Assumes `root` dir contains folder named `raw` with following files that contain results
        for tracing vertex trajectories, building graphs:
        - vtx_pos.npy : shape (Frames, Vertices, 2), positions of tri-cellular junctions.
        - edges_index.npy : shape (2, Edges), edge indices-- each column contains indices of source
                            and target vertices/nodes.
        - node2cell_index.npy : shape (2, Node-to-Cell edges), first row contains indices of the
                            nodes/vertices, and second row contains indices of the corresponding
                            cell IDs (cell indices).
        - edge_Length.npy (optional) : shape (Frames, Edges, 1), edge lengths.

        Arg-s:
        - root : path to a root directory that contains folder with raw dataset(s) in a folder named "raw".
        Raw datasets should be placed into separate folders each containing outputs from a single simulation.
        E.g. root contains ["raw", "processed", ...], and in folder "raw/" we should have ["simul1", "simul2", ...]
        - window_size : number of past velocities to be used as node features ordered from `T-window_size` to `T-1`
                       `[x(T-window_size+1)-x(T-window_size),..., x(T-1)-x(T-2), x(T+0)-x(T-1)]`, where `x(T)` is
                       node position at time `T`. For more details see method `VertexDynamics.pos2nodeXY()`.
        - transform :  transform(s) for graph datasets (e.g. from torch_geometric.transforms ), used in
                       parent class' loading method.
        - pre_transform : transform(s) for data pre-processing (resulting graphs are saved in "preprocessed" folder)
        and used as this dataset's sample graphs.

        Notes:
        - Velocities are approximated as 1st order differences of positions `x` (vtx_pos.npy) in subsequent frames:
          `velocity(T+0) = x(T+1) - x(T+0)`.
        - Use `pre_transform` for normalising and pre-processing dataset(s).
        '''
        super(HaraMovies, self).__init__(root, window_size=window_size, transform=transform,
                                         pre_transform=pre_transform, pre_filter=pre_filter,
                                         smoothing=smoothing, sma_lag_time=sma_lag_time)

    @property
    def processed_file_names(self):
        '''
        Return list of pytorch-geometric data files in `root/processed` folder (`self.processed_dir`).
        '''
        # "last_idx" : last index of window in "vertex velocity" (for features)
        # last_idx=T-(2+window_size) --> num of processed frames: num_of_frames=last_idx+1
        nums_of_frames = [(path.basename(raw_path),
                           load_array(path.join(raw_path, 'vtx_pos.npy')).shape[0] - (2+self.window_size) + 1
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
            # Load node positions from raw_path and convert to (windowed) node attrib-s and targets.
            node_pos, X_node, Y_node = self.pos2nodeXY(pos_path=path.join(raw_path, 'vtx_pos.npy'))

            # edge indices
            edge_index = torch.from_numpy(load_array(path.join(raw_path, 'edges_index.npy'))
                                          ).type(torch.long).contiguous()

            # cell-to-node and node-to-cell "edge indices"
            node2cell_index = torch.from_numpy(load_array(path.join(raw_path, 'node2cell_index.npy'))
                                               ).type(torch.long).contiguous()  # node_id-cell_id pairs
            cell2node_index = node2cell_index[[1, 0]].contiguous()  # cell_id-node_id pairs

            mov_name = path.basename(raw_path)  # folder name for the files
            N_nodes = node_pos.size(1)  # number of vertices/nodes (assume constant w.r.t. time)
            N_cells = cell2node_index[0].max().item() + 1  # num_of_cells (assume constant w.r.t. time)

            for t in range(node_pos.size(0)):
                data = CellData(num_nodes=N_nodes,
                                num_cells=N_cells,
                                edge_index=edge_index,
                                node2cell_index=node2cell_index, cell2node_index=cell2node_index,
                                pos=node_pos[t], x=X_node[t], y=Y_node[t],
                                cell_pressures=None,
                                edge_tensions=None
                                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, path.join(self.processed_dir, 'data_{}_{}.pt'.format(mov_name, t)))


class HaraAblation(VertexDynamics):
    '''
    For working with Y. Hara et al. amnioserosa ablation data. Hara ablation
    dataset *does not* have cell pressures, edge tensions, and T+0 velocities.
    Instead of edge tensions `HaraAblation` contains edge recoil results from
    laser ablation experiments.

    Overwrites following methods in `VertexDynamics`:
        - `processed_file_names`
        - `process`
        - `pos2nodeXY`
    '''

    def __init__(self, root, window_size=5, transform=None, pre_transform=None, pre_filter=None,
                 smoothing=False, sma_lag_time=None):
        '''
        Assumes `root` dir contains folder named `raw` with following files that contain results
        for tracing vertex trajectories, building graphs:
        - vtx_pos.npy : shape (Frames, Vertices, 2), positions of tri-cellular junctions.
        - edges_index.npy : shape (2, Edges), edge indices-- each column contains indices
                            of source and target vertices/nodes.
        - node2cell_index.npy : shape (2, Node-to-Cell edges), first row contains indices
                                of the nodes/vertices, and second row contains indices of
                                the corresponding cell IDs (cell indices).
        - edge_recoils.npy : recoil values for all edges. All values are "np.nan" except
                             for target edge(s), shape (Edges,). E.g.
                             `[nan,..., nan, 0.0238768, nan,...,nan]`.
        - frames.npy: (optional) frame numbers for the first dimension of array
                      in `vtx_pos.npy`, shape (Frames,).

        Arg-s:
        - root : path to a root directory that contains folder with raw dataset(s) in a
                 folder named "raw". Raw datasets should be placed into separate folders
                 each containing outputs from a single ablation sample.
        E.g. root contains ["raw", "processed", ...], and in folder "raw/" we should have
             ["sample_i", "sample_j", ...]
        - window_size : number of past velocities to be used as node features ordered from
                        `T-window_size` to `T-1`,
                        `[x(T-window_size+1)-x(T-window_size),..., x(T-1)-x(T-2), x(T+0)-x(T-1)]`,
                        where `x(T)` is node position at time `T`. For more details see method
                        `VertexDynamics.pos2nodeXY()`.
        - transform : transform(s) for graph datasets (e.g. from torch_geometric.transforms ),
                      used in parent class' loading method.
        - pre_transform : transform(s) for data pre-processing (resulting graphs are saved in
                          "preprocessed" folder) and used as this dataset's sample graphs.

        Notes:
        - Velocities are approximated as 1st order differences of positions `x` (vtx_pos.npy)
          in subsequent frames: `velocity(T+0) = x(T+1) - x(T+0)` (*only* for features, as T+0
          is absent in ablation data).
        - Use `pre_transform` for normalising and pre-processing dataset(s).
        - To be a valid sample in `HaraAblation` dataset, an ablation movie should have at
          least `window_size+1` frames, otherwise an example is ignored. `HaraAblation` uses
          the last `window_size+1` number of frames to produce `window_size` number of past
          velocities (`x`) and T+0 positions (`pos`).
        '''
        super(HaraAblation, self).__init__(root, window_size=window_size, transform=transform,
                                           pre_transform=pre_transform, pre_filter=pre_filter,
                                           smoothing=smoothing, sma_lag_time=sma_lag_time)

    @property
    def processed_file_names(self):
        '''
        Return list of pytorch-geometric data files in `root/processed` folder (`self.processed_dir`).
        '''
        # "last_idx" : last index of window in "vertex velocity" (for features)
        # last_idx=T-(2+window_size) --> num of processed frames: num_of_frames=last_idx+1
        # movie should have at least `window_size`+1 frames
        valid_data_names = [path.basename(raw_path) for raw_path in self.raw_paths
                            if load_array(path.join(raw_path, 'vtx_pos.npy')).shape[0] > self.window_size]
        file_names = ['{}.pt'.format(n) for n in valid_data_names]
        return file_names

    def process(self):
        '''
        Assumptions:
        - the parent class init runs _process() and initialises all the required dir-s.
        - cell graph topology and number of nodes is constant w.r.t. to frames.
        '''
        for raw_path in self.raw_paths:
            # Load node positions from raw_path and convert to (windowed) node attrib-s and targets.
            node_pos, X_node = self.pos2nodeXY(pos_path=path.join(raw_path, 'vtx_pos.npy'))

            # edge indices
            edge_index = torch.from_numpy(load_array(path.join(raw_path, 'edges_index.npy'))
                                          ).type(torch.long).contiguous()

            # edge recoils
            edge_recoils = torch.from_numpy(load_array(path.join(raw_path, 'edge_recoils.npy'))
                                            ).type(dtype)

            # cell-to-node and node-to-cell "edge indices"
            node2cell_index = torch.from_numpy(load_array(path.join(raw_path, 'node2cell_index.npy'))
                                               ).type(torch.long).contiguous()  # node_id-cell_id pairs
            cell2node_index = node2cell_index[[1, 0]].contiguous()  # cell_id-node_id pairs

            N_nodes = node_pos.size(1)  # number of vertices/nodes (assume constant w.r.t. time)
            N_cells = cell2node_index[0].max().item() + 1  # num_of_cells (assume constant w.r.t. time)

            data = CellData(num_nodes=N_nodes,
                            num_cells=N_cells,
                            edge_index=edge_index,
                            node2cell_index=node2cell_index, cell2node_index=cell2node_index,
                            pos=node_pos, x=X_node, y=None,
                            cell_pressures=None,
                            edge_tensions=None, edge_recoils=edge_recoils
                            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, path.join(self.processed_dir, '{}.pt'.format(path.basename(raw_path))))

    def pos2nodeXY(self, pos_path=None):
        '''
        Load 'simul_vtxpos.npy' and pre-process it into windowed data of velocities (1st differences)
        using `self.window_size` number of past velocities (frames) as node features.

        - pos_path: path to vertex positions file, e.g. 'vtx_pos.npy', containing an array with
                    shape `(frames)xNx2`.

        Returns: {dtype : `torch.float32`}
        - node_pos: node positions : shape Nx2
        - X_node: node velocities ordered from `T-window_size` to `T-1` : shape Nx(window_size)x2

        Where the `num_of_frames=last_idx+1`, and `last_idx=frames-(2+window_size)` is the last index
        of window in vx_vel (full array of 1st differences of the node positions with "frames-1"
        number of frames, i.e. vx_pos.shape[0]==frames).

        Note that `Y_node` the `T+0` vertex velocities are not available.
        '''
        # vertex trajectories:(Frames,Vertices,Dims)=TxNx2
        vx_pos = load_array(pos_path)  # TxNx2

        if self.smoothing:
            vx_pos = self.apply_smoothing(vx_pos)

        vx_vel = np.diff(vx_pos, n=1, axis=0)  # velocity(1st diff approx):(T-1)xNx2

        # T+0 vertex positions
        node_pos = torch.from_numpy(vx_pos[-1]).type(dtype)  # Nx2

        # last (T-window_size) to (T-1) velocities : Nx(window_size)x2
        X_node = torch.from_numpy(vx_vel[vx_vel.shape[0]-self.window_size:vx_vel.shape[0]].transpose((1, 0, 2))
                                  ).type(dtype)
        return node_pos, X_node
