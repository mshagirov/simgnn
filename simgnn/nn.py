import torch
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_scatter import scatter


class mlp(torch.nn.Module):
    '''
    MLP consisting of multiple linear layers w/ activation func-s (`Fn`). Last layer is always linear layer w/o activation.

    The last layer is a linear layer, i.e. it has no dropout/activation.
    '''

    def __init__(self, in_features, out_features, hidden_dims=[], dropout_p=0, Fn=ReLU, Fn_kwargs={}):
        '''
        - in_features : input dim-s.
        - out_features: output dim-s.
        - hidden_dims : a list of hidden dim-s (number of hidden layer neurons) {default : [] an empty list, i.e. no hidden layers}
        - dropout_p   : dropout prob-y for hidden layer(s) {default :  0}.
        - Fn : activation function for hidden layers { default: ReLU }
        - Fn_kwargs : keyword arg-s for `Fn` {default : an empty dict.}
        '''
        super(mlp, self).__init__()

        layers_in = [in_features] + hidden_dims # in_features for all layers

        layers = []
        for l, hn in enumerate(hidden_dims):
            layers.append( Linear( layers_in[l], hn)) #append first and hidden layers
            layers.append( Fn(**Fn_kwargs))
            if dropout_p:
                layers.append( Dropout(p=dropout_p))
        layers.append( Linear( layers_in[-1], out_features))

        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class IndependentBlock(torch.nn.Module):
    '''
    Layer w/ two independent node and edge MLPs that use the same `mlp_kwargs` for both MLPs.
    The last layers are always plain linear layers, i.e. they have no dropout/activations.

    x_enc, e_enc = IndependentBlock(x, e);
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[],**mlp_kwargs):
        '''
        Arg-s:
            - in_dims : number of input dimensions. Either an int or a dict of integers w/ keys "node" and "edge".
            - out_dims : number of output dimensions for encoder MLPs. Either an int or a dict of integers.
            - hidden_dims : a list of hidden dimensions {default : [] no hidden layers}, or a dict of lists.
            - mlp_kwargs : kwarg-s for MLPs.

        For `in_dims`, `out_dims` use integers, and for `hidden_dims` use a list in order to have same
        input/output/hidden dimensions for both node and edge MLPs.
        '''
        super(IndependentBlock, self).__init__()
        if type(in_dims)==dict:
            node_in = in_dims['node']
            edge_in = in_dims['edge']
        else:
            node_in = in_dims
            edge_in = in_dims

        if type(out_dims)==dict:
            node_out = out_dims['node']
            edge_out = out_dims['edge']
        else:
            node_out = out_dims
            edge_out = out_dims

        if type(hidden_dims)==dict:
            node_hidden_dims = hidden_dims['node']
            edge_hidden_dims = hidden_dims['edge']
        else:
            node_hidden_dims = hidden_dims
            edge_hidden_dims = hidden_dims

        self.node_mlp = mlp(node_in, node_out, hidden_dims=node_hidden_dims,**mlp_kwargs)
        self.edge_mlp = mlp(edge_in, edge_out, hidden_dims=edge_hidden_dims,**mlp_kwargs)

    def forward(self, x, edge_attr):
        return self.node_mlp(x), self.edge_mlp(edge_attr)


class Message(torch.nn.Module):
    '''Updates a graph's edge features by computing messages `mlp([x_src, x_tgt, edge_attr])--> new edge_attr`.'''

    def __init__(self, in_features, out_features, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : input dim-s == `#src_features` + `#tgt_features` + `#edge_features`.
        - out_features: output dim-s, e.g. `#edge_features`.

        Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(Message, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, src, tgt, edge_attr):
        '''
        - src, tgt : source and target features w/ shapes (#edges, #src_features) and (#edges, #tgt_features)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        return self.mlp( torch.cat( [src, tgt, edge_attr], 1) )


class DiffMessage(torch.nn.Module):
    '''
    Updates a graph's edge features by computing messages `mlp([ x_tgt - x_src, edge_attr])--> new edge_attr`.

    Uses differences in x_tgt - x_src rather than concatenating them.
    '''

    def __init__(self, in_features, out_features, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : input dim-s == `#src_features` + `#edge_features`. Assumes `#tgt_features`==`#src_features`.
        - out_features: output dim-s, e.g. `#edge_features`.

        Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(DiffMessage, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, src, tgt, edge_attr):
        '''
        - src, tgt : source and target features w/ shapes (#edges, #src_features) and (#edges, #tgt_features)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        return self.mlp( torch.cat( [tgt - src, edge_attr], 1) )


class AggregateUpdate(torch.nn.Module):
    '''Aggregates messages (`edge_attr`) from neighbouring nodes and updates node attributes.'''

    def __init__(self, in_features, out_features, aggr='mean', **mlp_kwargs):
        '''
        Arg-s:
        - in_features : input dim-s == `#node_features + #edge_features`. (MLP for updating x:node_features)
        - out_features: output dim-s, updated node fetaures, `#new_node_features`. (MLP for updating x:node_features)
        - aggr : aggregation scheme, one of `['sum', 'mul', 'mean', 'min', 'max']` {default: 'mean'}.

        Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(AggregateUpdate, self).__init__()
        assert aggr in ['sum', 'mul', 'mean', 'min', 'max']
        self.aggr = aggr
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, x, edge_index, edge_attr):
        '''
        - x : node features w/ shape (#nodes, #node_features)
        - edge_index : edge index (pairs of "src" and "tgt" indices) w/ shape (2, #edges)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        row, col = edge_index # (src, tgt) indices
        out = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce=self.aggr) # aggregate mssgs at "targets"
        out = torch.cat([x, out], dim=1) # concat w/ tgt node features
        out = self.mlp( out ) # update node features
        return out


class Aggregate(torch.nn.Module):
    '''Aggregates messages (`edge_attr`) from neighbouring nodes.'''

    def __init__(self, aggr='mean'):
        '''
        Arg-s:
        - aggr : aggregation scheme, one of `['sum', 'mul', 'mean', 'min', 'max']` {default: 'mean'}.
        '''
        super(Aggregate, self).__init__()
        assert aggr in ['sum', 'mul', 'mean', 'min', 'max']
        self.aggr = aggr

    def forward(self, dim_size, edge_index, edge_attr):
        '''
        - dim_size : number of output nodes :int
        - edge_index : edge index (pairs of "src" and "tgt" indices) w/ shape (2, #edges)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        out = scatter(edge_attr, edge_index[1], dim=0, dim_size=dim_size, reduce=self.aggr) # aggregate mssgs at "targets"
        return out


class Plain_MLP(torch.nn.Module):
    '''Simple MLP for processing pt-geometric graph vertex features `data.x`.

    Returns tuple (y_pred, None, None):
    - y_pred: is an output of `y_pred = MLP(data.x)`.
    - The two nones are just place holders to make MLP compatible with training function in `train.py`.
    '''
    def __init__(self, in_features=10, out_features=2, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : #input features
        - out_features: #output features
        - Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(Plain_MLP, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)
    def forward(self, data):
        return self.mlp( data.x ), None, None


class PlainSquaredMLP(torch.nn.Module):
    '''Simple MLP for processing pt-geometric graph vertex features `data.x`.

    Returns tuple (y_pred, None, None):
    - y_pred: is an output of `y_pred = PlainSquaredMLP(data.x)`. PlainSquaredMLP(x)=MLP([x,x^2])
    - The two nones are just place holders to make MLP compatible with training function in `train.py`.
    '''
    def __init__(self, in_features=10, out_features=2, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : #input features
        - out_features: #output features
        - Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(PlainSquaredMLP, self).__init__()
        self.mlp = mlp(in_features*2, out_features, **mlp_kwargs)
    def forward(self, data):
        return self.mlp( torch.cat([data.x,data.x**2],dim=1)), None, None


class Single_MP_step(torch.nn.Module):
    '''
    Returns tuple (y_pred, None, None):
    - y_pred: is a node-wise output (e.g. node velocity)
    - The two `None`s are just place holders to make model compatible with training function in `train.py`.
    '''
    def __init__(self, node_in_features=10, node_out_features=2, edge_in_features=2,
                 message_out_features=5, message_hidden_dims=[10], update_hidden_dims = [],aggr='mean', **mlp_kwargs):
        '''
        Arg-s:
        - node_in_features : #input node features
        - node_out_features: #output node features
        - edge_in_features : #input edge features
        - message_out_features : #message features (edge-wise messages, can be considered as new or intermediate edge features )
        - message_hidden_dims : list of #dims for message MLP=phi. For edge s->t: m_st = phi([x_t - x_s, e_st]).
        - update_hidden_dims : list of #dims for update MLP=gamma. For node i : x_i' = gamma(x_i, Aggregate(m_si))
        - Optional kwargs for both MLPs: defaults are `dropout_p = 0`, `Fn = ReLU`, `Fn_kwargs = {}`.
        '''
        super(Single_MP_step, self).__init__()

        self.message = DiffMessage(node_in_features+edge_in_features,
                                   message_out_features,
                                   hidden_dims=message_hidden_dims, **mlp_kwargs)
        self.relu = torch.nn.ReLU()
        self.aggr_update = AggregateUpdate(node_in_features+message_out_features,
                                           node_out_features, hidden_dims=update_hidden_dims, aggr=aggr, **mlp_kwargs)

    def forward(self, data):
        # convert to undirected graph : cat([e_ij, e_ji])
        edge_index = torch.cat([ data.edge_index, torch.stack([data.edge_index[1],
                                                               data.edge_index[0]], dim=0) ], dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr  = torch.cat([ data.edge_attr, -data.edge_attr], dim=0).contiguous()

        # message
        src, tgt = data.x[edge_index[0]], data.x[edge_index[1]] # src, tgt features
        m_ij = self.relu( self.message(src, tgt, edge_attr) )

        # aggregate and update stages
        x_out = self.aggr_update( data.x, edge_index, m_ij) # leave last layer as linear, i.e. no ReLU()
        return x_out, None, None


class DiffMessageSquared(torch.nn.Module):
    def __init__(self, in_features, out_features, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : input dim-s == 2*`#src_features` + `#edge_features`. Assumes `#tgt_features`==`#src_features`.
        - out_features: output dim-s, e.g. `#edge_features`.

        Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(DiffMessageSquared, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, src, tgt, edge_attr):
        '''
        - src, tgt : source and target features w/ shapes (#edges, #src_features) and (#edges, #tgt_features)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        return self.mlp( torch.cat( [tgt - src,(tgt - src)**2, edge_attr], dim=1) )


class SingleMPStepSquared(torch.nn.Module):
    def __init__(self, node_in_features=10, node_out_features=2, edge_in_features=2,
                 message_out_features=5, message_hidden_dims=[10], update_hidden_dims = [],aggr='mean', **mlp_kwargs):
        '''
        Arg-s:
        - node_in_features : #input node features
        - node_out_features: #output node features
        - edge_in_features : #input edge features
        - message_out_features : #message features (edge-wise messages, can be considered as new or intermediate edge features )
        - message_hidden_dims : list of #dims for message MLP=phi. For edge s->t: m_st = phi([x_t - x_s, e_st]).
        - update_hidden_dims : list of #dims for update MLP=gamma. For node i : x_i' = gamma(x_i, Aggregate(m_si))
        - Optional kwargs for both MLPs: defaults are `dropout_p = 0`, `Fn = ReLU`, `Fn_kwargs = {}`.
        '''
        super(SingleMPStepSquared, self).__init__()

        self.message = DiffMessageSquared(node_in_features*2+edge_in_features,
                                          message_out_features,
                                          hidden_dims=message_hidden_dims, **mlp_kwargs)
        self.relu = torch.nn.ReLU()
        self.aggr_update = AggregateUpdate(node_in_features*2+message_out_features,
                                           node_out_features, hidden_dims=update_hidden_dims, aggr=aggr, **mlp_kwargs)

    def forward(self, data):
        # convert to undirected graph : cat([e_ij, e_ji])
        edge_index = torch.cat([ data.edge_index, torch.stack([data.edge_index[1],
                                                               data.edge_index[0]], dim=0) ], dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr  = torch.cat([ data.edge_attr, -data.edge_attr], dim=0).contiguous()

        # message
        src, tgt = data.x[edge_index[0]], data.x[edge_index[1]] # src, tgt features
        m_ij = self.relu( self.message(src, tgt, edge_attr) )

        # aggregate and update stages
        x_out = self.aggr_update( torch.cat([data.x,data.x**2],dim=1),
                                 edge_index, m_ij) # leave last layer as linear, i.e. no ReLU()
        return x_out, None, None


class SingleMP_Tension(torch.nn.Module):
    '''
    Returns tuple (y_pred, None, None):
    - y_pred: is a node-wise output (e.g. node velocity)
    - The two `None`s are just place holders to make model compatible with training function in `train.py`.
    '''
    def __init__(self, node_in_features=10, node_out_features=2, edge_in_features=2,
                 message_out_features=5, message_hidden_dims=[10], update_hidden_dims = [],aggr='mean',
                 tension_out_features=1, tension_hidden_dims=[5], **mlp_kwargs):
        '''
        Arg-s:
        - node_in_features : #input node features
        - node_out_features: #output node features
        - edge_in_features : #input edge features
        - message_out_features : #message features (edge-wise messages, can be considered as new or intermediate edge features )
        - message_hidden_dims : list of #dims for message MLP=phi. For edge s->t: m_st = phi([x_t - x_s, e_st]).
        - update_hidden_dims : list of #dims for update MLP=gamma. For node i : x_i' = gamma(x_i, Aggregate(m_si))
        - Optional kwargs for both MLPs: defaults are `dropout_p = 0`, `Fn = ReLU`, `Fn_kwargs = {}`.
        '''
        super(SingleMP_Tension, self).__init__()

        self.message = DiffMessage(node_in_features+edge_in_features,
                                   message_out_features,
                                   hidden_dims=message_hidden_dims, **mlp_kwargs)
        self.relu = torch.nn.ReLU()
        self.aggr_update = AggregateUpdate(node_in_features+message_out_features,
                                           node_out_features, hidden_dims=update_hidden_dims, aggr=aggr, **mlp_kwargs)
        self.tension_mlp = mlp(message_out_features, tension_out_features,
                               hidden_dims=tension_hidden_dims, **mlp_kwargs)

    def forward(self, data):
        # convert to undirected graph : cat([e_ij, e_ji])
        edge_index = torch.cat([ data.edge_index, torch.stack([data.edge_index[1],
                                                               data.edge_index[0]], dim=0) ], dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr  = torch.cat([ data.edge_attr, -data.edge_attr], dim=0).contiguous()

        # message
        src, tgt = data.x[edge_index[0]], data.x[edge_index[1]] # src, tgt features
        m_ij = self.relu( self.message(src, tgt, edge_attr) )

        # aggregate and update stages
        x_out = self.aggr_update( data.x, edge_index, m_ij) # leave last layer as linear, i.e. no ReLU()

        # tension model
        e_out = self.tension_mlp(m_ij[:m_ij.size(0)//2,:] +  m_ij[m_ij.size(0)//2 :,:])

        return x_out, e_out.reshape((e_out.size(0),)), None
