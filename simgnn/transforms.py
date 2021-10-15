import torch


class AppendReversedEdges(object):
    '''
    Appends reversed (src-tgt --> tgt-src) edges to the graph. Optionally, reverses attributes (reverse is negative:
    e_st=-e_ts) and copies edge tensions (x_st=x_ts)
    '''
    def __init__(self, reverse_attr=False, reverse_tension=False):
        self.reverse_attr = reverse_attr
        self.reverse_tension = reverse_tension

    def __call__(self, data):
        data.edge_index = torch.cat([data.edge_index, torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)],
                                    dim=1).contiguous()
        if self.reverse_attr:
            data.edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0).contiguous()
        if self.reverse_tension:
            data.edge_tensions = torch.cat([data.edge_tensions, data.edge_tensions], dim=0).contiguous()
        return data

    def __repr__(self):
        return '{}(reverse_attr={}, reverse_tension={})'.format(self.__class__.__name__, self.reverse_attr,
                                                                self.reverse_tension)


class AppendEdgeDir(object):
    def __call__(self, data):
        data.edge_dir = data.edge_attr/torch.norm(data.edge_attr,dim=1,keepdim=True)
        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class Pos2Vec(object):
    '''Computes edge directions from connected node positions (source to target).'''
    def __init__(self, norm=True, scale=None, cat=False):
        '''
        Arg-s:
        - norm : if True, normalises/scales edge vectors (uses `scale` or maximum
                 component value if scale==None).
        - scale : scalar s.t. `scale>0`, edge vector componets are divided (scaled)
                 by this value.
        - cat : if True, concatenates edge vectors to edge attr-s, otherwise
                replaces current edge attr-s {default : False}.
        '''
        self.norm = norm
        self.scale = scale
        self.cat = cat

    def __call__(self, data):
        '''
        - data : input graphs (must contain node positions in `data.pos`)
        '''
        row, col = data.edge_index  # src, tgt indices

        e_vec = data.pos[col] - data.pos[row]  # src to tgt vectors

        if self.norm and (e_vec.numel() > 0):
            scale = e_vec.abs().max() if (self.scale is None) else self.scale
            e_vec = e_vec / scale if (scale > 0) else e_vec

        if data.edge_attr is not None and self.cat:
            data.edge_attr = data.edge_attr.view(-1, 1) if data.edge_attr.dim() == 1 else data.edge_attr
            data.edge_attr = torch.cat([data.edge_attr, e_vec.type_as(data.edge_attr)], dim=-1)
        else:
            data.edge_attr = e_vec

        return data

    def __repr__(self):
        return '{}(norm={}, scale={}, cat={})'.format(self.__class__.__name__, self.norm, self.scale, self.cat)


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
    `data.edge_tensions = ( data.edge_tensions )`.
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
        data.x = data.x.reshape(self.shape)
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
