import torch

class Pos2Vec(object):
    '''Computes edge directions from connected node positions (source to target).'''
    def __init__(self, norm=True, scale = None, cat=False):
        '''
        Arg-s:
        - norm : if True, normalises/scales edge vectors (uses `scale` or maximum component value if scale==None).
        - scale : scalar s.t. `scale>0`, edge vector componets are divided (scaled) by this value.
        - cat : if True, concatenates edge vectors to edge attr-s (default), otherwise replaces current edge attr-s.
        '''
        self.norm = norm
        self.scale = scale
        self.cat = cat
    
    def __call__(self, data):
        '''
        - data : input graphs (must contain node positions in `data.pos`)
        '''
        row, col = data.edge_index # src, tgt indices
        
        e_vec = data.pos[col] - data.pos[row] # src to tgt vectors
        
        if self.norm and (e_vec.numel() > 0):
            scale = e_vec.abs().max() if (self.scale is None) else self.scale
            e_vec = e_vec / scale if (scale>0) else e_vec
        
        if data.edge_attr is not None and self.cat:
            data.edge_attr = data.edge_attr.view(-1, 1) if data.edge_attr.dim() == 1 else data.edge_attr
            data.edge_attr = torch.cat([data.edge_attr, e_vec.type_as(data.edge_attr)], dim=-1)
        else:
            data.edge_attr = e_vec
        
        return data
    
    def __repr__(self):
        return '{}(norm={}, scale={}, cat={})'.format(self.__class__.__name__, self.norm, self.scale, self.cat)


class ScaleVelocity(object):
    '''Scales velocities (`data.x` and `data.y`) by a given amount : e.g. `data.x = data.x/scale`.'''
    def __init__(self, scale):
        '''
        Arg-s:
        - scale : a scalar s.t. `scale>0`, `data.x` and `data.y` are divided by `scale`.
        '''
        assert scale>0
        self.scale = scale
    
    def __call__(self, data):
        '''
        - data : an input graph.
        '''
        if data.x is not None:
            data.x = data.x/self.scale
        
        if data.y is not None:
            data.y = data.y/self.scale
        
        return data
    
    def __repr__(self):
        return '{}(scale={})'.format(self.__class__.__name__, self.scale)

