import torch


class AppendReversedEdges(object):
    '''
    Appends reversed (src-tgt --> tgt-src) edges to the graph. Optionally, reverses attributes (reverse is negative:
    e_st=-e_ts) and copies edge tensions (x_st=x_ts)
    '''
    def __init__(self, reverse_attr=True, reverse_tension=False, edge_id=True):
        '''
        Arg-s:
        - reverse_attr : if true, appends negative copy of edge attributes `cat([edge_attr, -edge_attr])-->edge_attr`.
        - reverse_tension : if true, appends copy of edge tensions `cat([edge_tensions,edge_tensions])-->edge_tensions`.
        - edge_id : if true, adds new graph variable `edge_id` with int IDs for edges.
        '''
        self.reverse_attr = reverse_attr
        self.reverse_tension = reverse_tension
        self.edge_id = edge_id

    def __call__(self, data):
        if self.edge_id:
            data.edge_id = torch.cat([torch.arange(data.num_edges), torch.arange(data.num_edges)], dim=0).contiguous()
        if self.reverse_attr:
            data.edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0).contiguous()
        if self.reverse_tension:
            data.edge_tensions = torch.cat([data.edge_tensions, data.edge_tensions], dim=0).contiguous()

        data.edge_index = torch.cat([data.edge_index,
                                    torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)], dim=1).contiguous()
        return data

    def __repr__(self):
        return '{}(reverse_attr={}, reverse_tension={}, edge_id={})'.format(self.__class__.__name__, self.reverse_attr,
                                                                            self.reverse_tension, self.edge_id)


class AppendEdgeNorm(object):
    '''
    Appends norms of the first N_dim columns of `data.edge_attr` vectors to the end of `data.edge_attr`.

    `data.edge_attr = cat([data.edge_attr, norm(data.edge_attr[:,[0,1]])])`
    '''
    def __init__(self, N_dim=2):
        self.N_dim = N_dim
        self.cols = list(range(N_dim))

    def __call__(self, data):
        # calculate norm using first two col-s of data.edge_attr
        data.edge_attr = torch.cat([data.edge_attr,
                                    torch.linalg.norm(data.edge_attr[:, self.cols], dim=1, keepdim=True)
                                    ], dim=-1).contiguous()
        return data

    def __repr__(self):
        return '{}(N_dim={})'.format(self.__class__.__name__, self.N_dim)


class AppendDiff_x(object):
    '''
    Appends x_t-x_s and their Euclidean norms (optional, if `norm=True`).
    '''
    def __init__(self, norm=True):
        '''Appends Euclidean if `norm=True`'''
        self.norm = norm

    def __call__(self, data):
        # src to tgt vectors x_t-x_s; Nx(window_size)x2
        e_vec = data.x[data.edge_index[1]] - data.x[data.edge_index[0]]

        # append norms of x_t-x_s
        if self.norm:
            e_vec = torch.cat([e_vec, torch.linalg.norm(e_vec, dim=-1, keepdim=True)], dim=-1)

        # flatten and append to data.edge_attr
        data.edge_attr = torch.cat([data.edge_attr,
                                    e_vec.reshape(e_vec.size(0), -1)], dim=-1).contiguous()
        return data

    def __repr__(self):
        return '{}(norm={})'.format(self.__class__.__name__, self.norm)


class AppendEdgeLen(object):
    '''
    Computes edge lengths, and optionally directions, then appends them as
    a new graph variable(s) `data.edge_length` (, `data.edge_dir`). Directions
    are unit vectors from `src` to `tgt` vertices, i.e. `src, tgt = data.edge_index`.

    Agr-s:
        keep_dir : if true, appends computed edge directions as `data.edge_dir` (doesn't use ).
        aggr_edge_id : if true and `edge_id!=None`uses edge ids to compute a single direction
                       (src-tgt) for edge lengths since both direction (src-tgt, tgt-src) have
                       the same lengths.
        use_edge_attr : if true uses `data.edge_attr` as edge vectors instead of computing them from `pos`.
    '''
    def __init__(self, keep_dir=False, aggr_edge_id=True, use_edge_attr=False,
                 norm=False, scale=None):
        self.keep_dir = keep_dir
        self.aggr_edge_id = aggr_edge_id
        self.use_edge_attr = use_edge_attr
        self.norm = norm
        self.scale = scale

    def __call__(self, data):
        # compute edge vectors
        if self.use_edge_attr:
            e_vec = data.edge_attr
        else:
            row, col = data.edge_index  # src, tgt indices
            e_vec = data.pos[col] - data.pos[row]  # src to tgt vectors

        # normalise if `norm` is True
        if self.norm and (e_vec.numel() > 0):
            scale = e_vec.abs().max() if (self.scale is None) else self.scale
            e_vec = e_vec / (scale+10**-9)

        edge_length = torch.linalg.norm(e_vec, dim=1, keepdim=True)

        if self.aggr_edge_id and (data.edge_id is not None):
            data.edge_length = edge_length[torch.unique(data.edge_id)].reshape(-1,)
            if self.keep_dir:
                # compute unit vectors for edge directions
                data.edge_dir = e_vec[torch.unique(data.edge_id)]/edge_length[torch.unique(data.edge_id)]
        else:
            data.edge_length = edge_length
            if self.keep_dir:
                # compute unit vectors for edge directions
                data.edge_dir = e_vec/edge_length

        return data

    def __repr__(self):
        return '{}(keep_dir={}, aggr_edge_id={}, use_edge_attr={}, norm={}, scale={})'.format(
            self.__class__.__name__, self.keep_dir, self.aggr_edge_id, self.use_edge_attr,
            self.norm, self.scale)


class Pos2Vec(object):
    '''Computes edge directions from connected node positions (source to target).'''
    def __init__(self, norm=True, scale=None, cat=False, pos_noise=None,
                 noise_args=[], noise_kwargs={}):
        '''
        Arg-s:
        - norm : if True, normalises/scales edge vectors (uses `scale` or maximum
                 component value if scale==None).
        - scale : scalar s.t. `scale>0`, edge vector componets are divided (scaled)
                 by this value.
        - cat : if True, concatenates edge vectors to edge attr-s, otherwise
                replaces current edge attr-s {default : False}.
        - pos_noise : a function for producing an additive position noise (e.g. torch.normal);
                      must accept "size=" kwarg.
        '''
        self.norm = norm
        self.scale = scale
        self.cat = cat
        self.pos_noise = pos_noise
        self.noise_args = noise_args
        self.noise_kwargs = noise_kwargs

    def __call__(self, data):
        '''
        - data : input graphs (must contain node positions in `data.pos`)
        '''
        row, col = data.edge_index  # src, tgt indices
        pos = data.pos.detach().clone()

        if self.pos_noise != None:
            pos = pos + self.pos_noise(*self.noise_args,
                                       **self.noise_kwargs,
                                       size=pos.size())
            
        e_vec = pos[col] - pos[row]  # src to tgt vectors

        if self.norm and (e_vec.numel() > 0):
            scale = e_vec.abs().max() if (self.scale is None) else self.scale
            e_vec = e_vec / scale if (scale > 0) else e_vec

        if (data.edge_attr != None) and self.cat:
            data.edge_attr = data.edge_attr.view(-1, 1).contiguous() if data.edge_attr.dim() == 1 else data.edge_attr
            data.edge_attr = torch.cat([data.edge_attr, e_vec.type_as(data.edge_attr)], dim=-1).contiguous()
        else:
            data.edge_attr = e_vec

        return data

    def __repr__(self):
        params_list = [self.norm, self.scale, self.cat,
                       self.pos_noise.__name__ if self.pos_noise != None else self.pos_noise,
                       self.noise_args, self.noise_kwargs]
            
        return '{}(norm={}, scale={}, cat={}, pos_noise={}, noise_args={}, noise_kwargs={})'.format(self.__class__.__name__,*params_list)


class ScaleVar(object):
    '''
    Abstract class for scaling variables `data.var` by a given amount s.t. `data.var = data.var/scale`.
    After inheriring `ScaleVar` you will need to write the `__call__`, and optionally the `__repr__` methods.
    '''
    def __init__(self, scale):
        '''
        Arg-s:
        - scale : a scalar s.t. `scale>0`, `data.x` and `data.y` are divided by `scale`.
        '''
        assert scale > 0
        self.scale = scale

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        pass

    def __repr__(self):
        return '{}(scale={})'.format(self.__class__.__name__, self.scale)


class TransformVar(object):
    '''
    Abstract class for applying non-linear transformations on variables `data.var` w/ a given transformation T
    s.t. `data.var = self.T(data.var)`. After inheriring `TransformVar` you will need to write the `__call__`,
    and optionally the `__repr__` methods.
    '''
    def __init__(self, T):
        '''
        Arg-s:
        - t_func : a torch function that can accept `data.var`--a torch tensor as an input, e.g. `torch.log`.
        '''
        self.T = T

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        pass

    def __repr__(self):
        return '{}(T={}())'.format(self.__class__.__name__, self.T.__name__)


class ScaleVelocity(ScaleVar):
    '''Scales velocities (`data.x` and `data.y`) by a given amount : e.g. `data.x = data.x/scale`.'''
    def __init__(self, scale):
        '''
        Arg-s:
        - scale : a scalar s.t. `scale>0`, `data.x` and `data.y` are divided by `scale`.
        '''
        super(ScaleVelocity, self).__init__(scale)

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        if data.x is not None:
            data.x = data.x/self.scale

        if data.y is not None:
            data.y = data.y/self.scale

        return data


class ScaleTension(ScaleVar):
    '''
    Scales tension in `data.edge_tensions` by a given amount:
    `data.edge_tensions = ( data.edge_tensions - shift )/scale`.
    '''
    def __init__(self, scale, shift=0):
        '''
        Arg-s:
        - scale : must be `scale>0`.
        - shift : mean shift {default: 0}.
        '''
        super(ScaleTension, self).__init__(scale)
        self.shift = shift

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        if data.edge_tensions is not None:
            data.edge_tensions = (data.edge_tensions - self.shift)/self.scale
        return data

    def __repr__(self):
        return '{}(scale={}, shift={})'.format(self.__class__.__name__, self.scale, self.shift)


class ScalePressure(ScaleTension):
    '''
    Scales cell pressures in `data.cell_pressures` by a given amount:
    `data.cell_pressures = ( data.cell_pressures - shift )/scale`.
    '''
    def __init__(self, scale, shift=0):
        '''
        Arg-s:
        - scale : must be `scale>0`.
        - shift : mean shift {default: 0}.
        '''
        super(ScalePressure, self).__init__(scale, shift=shift)

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        if data.cell_pressures is not None:
            data.cell_pressures = (data.cell_pressures - self.shift)/self.scale
        return data


class TransformTension(TransformVar):
    '''
    Applies a transformation T on tension in `data.edge_tensions`:
    `data.edge_tensions = T( data.edge_tensions )`.
    '''
    def __init__(self, T):
        '''
        Arg-s:
        - T : a valid transformation function.
        '''
        super(TransformTension, self).__init__(T)

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        if data.edge_tensions is not None:
            data.edge_tensions = self.T(data.edge_tensions)
        return data


class Reshape_x(object):
    '''
    Reshapes attribute `x` (e.g. data.x) using a given shape.
    '''
    def __init__(self, shape):
        '''
        Arg-s:
        - shape : new shape for attribute `x`.
        '''
        self.shape = shape

    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        data.x = data.x.reshape(self.shape).contiguous()
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.shape)


class RecoilAsTension(object):
    '''
    If present, copies edge recoil values in `data.edge_recoils` to `data.edge_tensions` variable.
    '''
    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        if data.edge_recoils is not None:
            data.edge_tensions = data.edge_recoils.detach().clone()
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
