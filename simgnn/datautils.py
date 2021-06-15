__all__ = ['write_array', 'load_array', 'write_graph', 'load_graph', 'mknewdir']
import numpy as np
import pickle
from os import mkdir


def simple_moving_average(vtxpos,lag_time=4):
    '''
    Compute simple moving average along time axis, which is assumed to be the first axis (axis=0).
    
    - lag_time : lag time for past values, a number of past values to use. Moving average at
                 time `t` uses `lagtime+1` values including the current value x[t], i.e. average of
                 `{ x[t-lagtime], x[t+1-lagtime], ..., x[t]}`.
    '''
    vtxpos_sma = np.zeros_like(vtxpos)
    vtxpos_sma[0] = vtxpos[0]
    for t in range(1,vtxpos.shape[0]):
        vtxpos_sma[t] = vtxpos[max([t-lag_time, 0]):t+1].mean(axis=0)
    return vtxpos_sma


def mknewdir(dirpath):
    '''mkdir but prints that dir already exists instead of throwing exception.
    - Returns 1 (int) if dir already exists, or returns None otherwise.
    '''
    try:
        mkdir(dirpath)
    except FileExistsError:
        print('Directory already exists.')
        return 1


# functions for writing/reading np arrays to/from files
def write_array(fpath, arr):
    '''
    Write numpy array to file using `np.save`.

    Arg-s:
    - fpath: path to file, e.g. `array.npy`
    - arr: numpy nd-array.
    '''
    with open(fpath, 'wb') as f:
        np.save(f, arr)


def load_array(fpath):
    '''
    Load numpy array from file (`fpath`) using `np.load`.
    '''
    with open(fpath, 'rb') as f:
        arr = np.load(f)
    return arr


# functions for writing/reading dict [edges and cells] to/from files
def write_graph(fpath, m):
    '''
    Write cell monolayer's graph and cells as a pickle file. This does not include the vertices.
    Arg-s:
    - m: cell monolayer object.
    - fpath: location to write the file, e.g. 'graph_dict.pkl'.
    Edges and and cells are saved as a dictionary object with keys 'edges' and 'cells'.
    '''
    with open(fpath, 'wb') as f:
        pickle.dump({'edges':m.edges.tolist(), 'cells':m.cells}, f)


def load_graph(fpath):
    '''Load dictionary of cells and edges from a pickle file ('*.pkl').'''
    with open(fpath, 'rb') as f:
        graph_dict = pickle.load(f)
    return graph_dict
