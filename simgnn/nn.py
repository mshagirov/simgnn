import torch
from torch.nn import ModuleDict, ModuleList, Sequential, Linear, ReLU, Dropout
from torch_scatter import scatter
from collections import OrderedDict


class mlp(torch.nn.Module):
    '''
    MLP consisting of multiple linear layers w/ activation func-s (`Fn`). Last
    layer is always linear layer w/o activation.

    The last layer is a linear layer, i.e. it has no dropout/activation.
    '''

    def __init__(self, in_features, out_features, hidden_dims=[], dropout_p=0,
                 Fn=ReLU, Fn_kwargs={}):
        '''
        - in_features : input dim-s.
        - out_features: output dim-s.
        - hidden_dims : a list of hidden dim-s (number of hidden layer neurons)
                        {default : [] an empty list, i.e. no hidden layers}
        - dropout_p   : dropout prob-y for hidden layer(s) {default :  0}.
        - Fn : activation function for hidden layers { default: ReLU }
        - Fn_kwargs : keyword arg-s for `Fn` {default : an empty dict.}
        '''
        super(mlp, self).__init__()

        layers_in = [in_features] + hidden_dims  # in_features for all layers

        layers = []
        for itm, hn in enumerate(hidden_dims):
            layers.append(Linear(layers_in[itm], hn))  # append first & hid-ns
            layers.append(Fn(**Fn_kwargs))
            if dropout_p:
                layers.append(Dropout(p=dropout_p))
        layers.append(Linear(layers_in[-1], out_features))

        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SelectiveActivation(torch.nn.Module):
    '''
    Apply given activation function Fn on var_id`th input variable(s).

    Arg-s:
        - var_id : index of input variable, can be an integer or a list
                   of integers.

    E.g.:
        Fn = SelectiveActivation(var_id=2)
        y1, y2, y3  = Fn(x1, x2, x3)
    is equivalent to
        y1, y2, y3 = x1, x2, ReLU(x3)
    '''
    def __init__(self, var_id=0, Fn=ReLU, Fn_kwargs={}):
        super(SelectiveActivation, self).__init__()
        self.Fn = Fn(**Fn_kwargs)
        self.var_ids = var_id if type(var_id) == list else [var_id]

    def forward(self, *vars_in):
        vars_in = list(vars_in)
        for var_id in self.var_ids:
            vars_in[var_id] = self.Fn(vars_in[var_id])
        return tuple(vars_in)


def dims_to_dict(*mlp_dims):
    '''
    Converts/broadcasts MLP dimension arg-s `mlp_dims` (int, dict, OrderedDict)
    to a set of OrderedDict's with same keys (keys represent graph variables).
    Keys of the dict/OrderedDict input arg-s in `mlp_dims` are used if any of
    the input arg-s is a dict or OrderedDict, or default ("node", "edge") keys
    are used otherwise. Input arg-s must have the same keys (and same ordering)
    if more than one of the `mlp_dims` are dict/OrderedDict's. For python
    versions earlier than v3.7 use OrderedDict in order to retain order
    of the dictionary keys.
    '''
    def is_dict(d):
        return True if type(d) == dict or type(d) == OrderedDict else False

    n_vars = max([(k, len(mlp_dim) if is_dict(mlp_dim) else 1)
                  for k, mlp_dim in enumerate(mlp_dims)], key=lambda x: x[1])

    var_names = ("node", "edge") if n_vars[1] == 1 \
        else tuple(mlp_dims[n_vars[0]].keys())

    mlp_dims_out = (
        OrderedDict(
                    ((var_k, mlp_dim[var_k] if is_dict(mlp_dim) else mlp_dim)
                        for var_k in var_names)) for mlp_dim in mlp_dims
                    )
    return tuple(mlp_dims_out)


class Message(torch.nn.Module):
    '''
    Concatenates and processes a list of input_tensors (must have same batch
    sizes, axis=0). `y=MLP(torch.cat( [*input_tensors], 1))`.
    '''
    def __init__(self, in_features, out_features, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : (sum of) input dim-s ==
                   `#src_features` + `#tgt_features` + `#edge_features` + ... .
        - out_features: output dim-s, e.g. `#edge_features`.

        Optional kwargs for `mlp`:
                     hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(Message, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, *input_tensors):
        '''`y=MLP(torch.cat( [*input_tensors], 1))`'''
        return self.mlp(torch.cat([*input_tensors], 1))


class DiffMessage(torch.nn.Module):
    '''
    Updates a graph's edge features by computing messages
    `mlp([ x_tgt - x_src, edge_attr])--> new edge_attr`.

    Uses differences in x_tgt - x_src rather than concatenating them.
    '''
    def __init__(self, in_features, out_features, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : input dim-s == `#src_features` + `#edge_features`.
        Assumes `#tgt_features`==`#src_features`.
        - out_features: output dim-s, e.g. `#edge_features`.

        Optional kwargs for `mlp`:
                hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(DiffMessage, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, src, tgt, edge_attr):
        '''
        - src, tgt : source and target features w/ shapes
                     (#edges, #src_features) and (#edges, #tgt_features)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        return self.mlp(torch.cat([tgt - src, edge_attr], 1))


class AggregateUpdate(torch.nn.Module):
    '''
    Aggregates edge features (`edge_attr`) from neighbouring nodes and updates node
    features.
    '''
    def __init__(self, in_features, out_features, aggr='mean', **mlp_kwargs):
        '''
        Arg-s:
        - in_features : input dim-s ==
          `#node_features + #edge_features`. (MLP for updating x:node_features)
        - out_features: output dim-s, updated node fetaures,
                       `#new_node_features`. (MLP for updating x:node_features)
        - aggr : aggregation scheme, one of `['sum', 'mul', 'mean', 'min',
                 'max']` {default: 'mean'}.

        Optional kwargs for `mlp`:
                     hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(AggregateUpdate, self).__init__()
        assert aggr in ['sum', 'mul', 'mean', 'min', 'max']
        self.aggr = aggr
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, x, edge_index, edge_attr):
        '''
        - x : node features w/ shape (#nodes, #node_features)
        - edge_index : edge index (pairs of "src" and "tgt" indices) w/
                       shape (2, #edges)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        row, col = edge_index  # (src, tgt) indices
        out = scatter(edge_attr, col, dim=0, dim_size=x.size(0),
                      reduce=self.aggr)  # aggregate mssgs at "targets"
        out = torch.cat([x, out], dim=1)  # concat w/ tgt node features
        out = self.mlp(out)  # update node features
        return out


class Aggregate(torch.nn.Module):
    '''Aggregates edge features (`edge_attr`) from neighbouring nodes.'''

    def __init__(self, aggr='mean'):
        '''
        Arg-s:
        - aggr : aggregation scheme, one of `['sum', 'mul', 'mean', 'min',
                'max']` {default: 'mean'}.
        '''
        super(Aggregate, self).__init__()
        assert aggr in ['sum', 'mul', 'mean', 'min', 'max']
        self.aggr = aggr

    def forward(self, dim_size, edge_index, edge_attr):
        '''
        - dim_size : number of output nodes :int
        - edge_index : edge index (pairs of "src" and "tgt" indices) w/
                       shape (2, #edges)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        out = scatter(edge_attr, edge_index[1], dim=0, dim_size=dim_size,
                      reduce=self.aggr)  # aggregate mssgs at "targets"
        return out


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
        return self.mlp(torch.cat([tgt - src, (tgt - src)**2, edge_attr], dim=1))


class NodeUpdate(torch.nn.Module):
    def __init__(self, f_node):
        super(NodeUpdate, self).__init__()
        self.f_node = f_node  # aggr+update nodes

    def forward(self, x, edge_index, edge_attr):
        # h_v = f_node(x, aggr(edge_attr|edge_index))
        h_v = self.f_node(x, edge_index, edge_attr)
        return h_v, edge_index, edge_attr


class EdgeUpdate(torch.nn.Module):
    def __init__(self, f_edge):
        super(EdgeUpdate, self).__init__()
        self.f_edge = f_edge  # aggr+update nodes

    def forward(self, x, edge_index, edge_attr):
        # h_e[k] = f_edge(x_s[k], x_t[k], e[k])
        h_e = self.f_edge(x[edge_index[0]],
                          x[edge_index[1]],
                          edge_attr)
        return x, edge_index, h_e


class SequentialUpdate(torch.nn.Module):
    def __init__(self, *f_vars):
        super(SequentialUpdate, self).__init__()
        self.updates = ModuleList(f_vars)

    def forward(self, x, edge_index, edge_attr):
        h_v, h_e = x, edge_attr
        for layer in self.updates:
            h_v, _, h_e = layer(h_v, edge_index, h_e)
        return h_v, edge_index, h_e


class ParallelUpdate(torch.nn.Module):
    def __init__(self, f_node, f_edge):
        super(ParallelUpdate, self).__init__()
        self.f_node = f_node  # aggr+update nodes
        self.f_edge = f_edge  # message/update edges

    def forward(self, x, edge_index, edge_attr):
        # h_v = f_node(x, aggr(edge_attr|edge_index))
        h_v = self.f_node(x, edge_index, edge_attr)

        # h_e[k] = f_edge(x_s[k], x_t[k], e[k])
        h_e = self.f_edge(x[edge_index[0]],
                          x[edge_index[1]],
                          edge_attr)
        return h_v, edge_index, h_e


class MessageBlock(torch.nn.Module):
    '''
    Vanilla message passing block with one aggregation and one update function per
    graph variable. The sequence of aggregate+update steps for variables {node, edge} can be
    one of {"e": edge-then-node, "n": node-then-edge, or "p": parallel/simulataneous}
    update schemes given by `updt` argument.

    Notes:
        - Aggregation of nodes to edges is fixed to concatenation operation, whereas
        aggregation of edges to nodes can be one of the schemes set by an input argument `aggr`.
        - Update function can be a linear layer or a MLP. For all cases, the last layer is always
        linear.
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[],
                 aggr='mean', updt='e', **mlp_kwargs):
        '''
        Arg-s:
            - in_dims, out_dims : number of input and output dimensions. Either an
                      int (if all MLPs have same input/output dim-s) or a dict of
                      integers w/ keys "node" and "edge", e.g. {'node':10,'edge':2, ...}.
            - hidden_dims : a list of hidden dimensions {default : [] no hidden
                      layers}, or a dict of lists, e.g. {'node':[],'edge':[8,16], ...}.
            - aggr : message aggregation scheme, one of `['sum', 'mul', 'mean', 'min', 'max']`
                    {default: 'mean'}.
            - updt : update sequence, one of 'e' (edge-then-node), 'n' (node-then-edge),
                    or 'p' (parallel or simulataneous update).
            - mlp_kwargs : optional kwarg-s for MLPs.

        Notes:
            - Dimensions of inputs for actual node and edge MLPs depend on update
              sequence `updt`.
            - In order to have same output or hidden dimensions for all
              MLPs use integers for `out_dims`, and a list for
              `hidden_dims`, instead of dictionaries.
        '''
        super(MessageBlock, self).__init__()
        assert updt in 'enp'

        in_dims, out_dims, hidden_dims = dims_to_dict(in_dims, out_dims, hidden_dims)

        if updt == 'e':
            # update edge-then-node {default}
            f_v_in = in_dims['node'] + out_dims['edge']
            f_e_in = in_dims['edge'] + 2*in_dims['node']
            seq = ['edge', 'node']  # update order
        elif updt == 'n':
            # update node-then-edge
            f_v_in = in_dims['node'] + in_dims['edge']
            f_e_in = in_dims['edge'] + 2*out_dims['node']
            seq = ['node', 'edge']  # update order
        elif updt == 'p':
            # parallel update
            f_v_in = in_dims['node'] + in_dims['edge']
            f_e_in = in_dims['edge'] + 2*in_dims['node']

        f_var = {'node': AggregateUpdate(f_v_in, out_dims['node'], hidden_dims=hidden_dims['node'],
                                         aggr=aggr, **mlp_kwargs),
                 'edge': Message(f_e_in, out_dims['edge'], hidden_dims=hidden_dims['edge'],
                                 **mlp_kwargs)}

        if updt == 'p':
            self.layers = ParallelUpdate(f_var['node'], f_var['edge'])
        else:
            f_update = {'node': NodeUpdate, 'edge': EdgeUpdate}
            self.layers = SequentialUpdate(*(f_update[k](f_var[k]) for k in seq))

    def forward(self, x, edge_index, edge_attr):
        '''
        Returns: h_v, edge_index, h_e
        '''
        h_v, edge_index, h_e = self.layers(x, edge_index, edge_attr)
        return h_v, edge_index, h_e


class ResidualMessageBlock(MessageBlock):
    '''`MessageBlock` with residual (skip) connnection `output=x+MessageBlock(x)`'''
    def __init__(self, in_dims, out_dims, hidden_dims=[],
                 aggr='mean', updt='e', **mlp_kwargs):
        super(ResidualMessageBlock, self).__init__(in_dims, out_dims,
                                                   hidden_dims=hidden_dims,
                                                   aggr=aggr, updt=updt,
                                                   **mlp_kwargs)

    def forward(self, x, edge_index, edge_attr):
        '''
        h_v, h_e = (x_v, x_e) + MessageBlock(x_v, x_e)

        Returns:
            h_v, edge_index, h_e
        '''
        h_v, edge_index, h_e = self.layers(x, edge_index, edge_attr)
        h_v = h_v + x  # node features
        h_e = h_e + edge_attr  # edge features
        return h_v, edge_index, h_e


class IndependentBlock(torch.nn.Module):
    '''
    Layer w/ independent MLPs that all use the same `mlp_kwargs`. The last
    layer of all MLPs are plain linear layers, i.e. they have no dropout/
    activations.

    Example:
        # Inputs --> x: [#nodes, 10]; e: [#edges, 2], ...
        # Enc : encodes node and edge features to 32-dim vectors (MLPs w/ one
        16-dim hidden layer)
        Enc = IndependentBlock({'node':10,'edge':2}, 32, hidden_dims=[16])
        x_enc, e_enc, ... = Enc(x, e, ...);
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[], **mlp_kwargs):
        '''
        Arg-s:
            - in_dims, out_dims : number of input and output dim-s. Either an
              int (if all MLPs have same input/output dim-s) or a dict of
              integers w/ keys "node", "edge", etc..
              E.g. {'node':10,'edge':2, ...}
            - hidden_dims : a list of hidden dimensions {default : [] no hidden
            layers}, or a dict of lists. E.g. {'node':[],'edge':[8,16], ...}
            - mlp_kwargs : kwarg-s for MLPs.

        Notes:
            - In order to have same input, output or hidden dimensions for all
              MLPs use integers for `in_dims`, `out_dims`, and a list for
              `hidden_dims`, instead of dictionaries.
            - Order of the dict keys is used for an input order for the forward
              function. If more than one of the input arguments ( in_dims,
              out_dims, hidden_dims) are dict, it is assumed that they all have
              the same keys and ordering for keys. Default order for dict keys,
              when using an `int` for `in_dims`, `out_dims`, and a `list` for
              `hidden_dims` is ['node', 'edge'] (MLPs have same dimensions).
              For python versions <3.7 use int or `OrderedDict` for consistent
              order of the dictionary keys. For further details see doc and
              code for `simgnn.nn.dims_to_dict`.
        '''
        super(IndependentBlock, self).__init__()

        in_dims, out_dims, hidden_dims = dims_to_dict(in_dims, out_dims,
                                                      hidden_dims)

        self.mlp_dict = ModuleDict({k: mlp(in_dims[k], out_dims[k],
                                           hidden_dims=hidden_dims[k],
                                           **mlp_kwargs) for k in in_dims})

    def forward(self, *xs):
        ys = []
        for x, k in zip(xs, self.mlp_dict):
            ys.append(self.mlp_dict[k](x))
        return tuple(ys)


class SingleMPStepSquared(torch.nn.Module):
    def __init__(self, node_in_features=10, node_out_features=2, edge_in_features=2,
                 message_out_features=5, message_hidden_dims=[10], update_hidden_dims=[],
                 aggr='mean', **mlp_kwargs):
        '''
        Arg-s:
        - node_in_features : #input node features
        - node_out_features: #output node features
        - edge_in_features : #input edge features
        - message_out_features : #message features (edge-wise messages, can be considered as
                                 new or intermediate edge features )
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
        edge_index = torch.cat([data.edge_index,
                                torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)],
                               dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0).contiguous()

        # message
        src, tgt = data.x[edge_index[0]], data.x[edge_index[1]]  # src, tgt features
        m_ij = self.relu(self.message(src, tgt, edge_attr))

        # aggregate and update stages
        x_out = self.aggr_update(torch.cat([data.x, data.x**2], dim=1),
                                 edge_index, m_ij)  # leave last layer as linear, i.e. no ReLU()
        return x_out, None, None


class Plain_MLP(torch.nn.Module):
    '''Simple MLP for processing pt-geometric graph vertex features `data.x`.

    Returns tuple (y_pred, None, None):
    - y_pred: is an output of `y_pred = MLP(data.x)`.
    - The two nones are just place holders to make MLP compatible with training
      function in `train.py`.
    '''
    def __init__(self, in_features=10, out_features=2, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : #input features
        - out_features: #output features
        - Optional kwargs for `mlp`:
                     hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(Plain_MLP, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, data):
        return self.mlp(data.x), None, None


class PlainSquaredMLP(torch.nn.Module):
    '''Simple MLP for processing pt-geometric graph vertex features `data.x`.

    Forward function returns tuple (y_pred, None, None):
    - y_pred: is an output of `y_pred = PlainSquaredMLP(data.x)`.
              PlainSquaredMLP(x)=MLP([x,x^2])
    - The two `None`s are place holders to make MLP compatible with training
      function in `train.py`.
    '''
    def __init__(self, in_features=10, out_features=2, **mlp_kwargs):
        '''
        MLP Arg-s:
        - in_features : #input features
        - out_features: #output features
        - Optional kwargs for `mlp`:
                    hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(PlainSquaredMLP, self).__init__()
        self.mlp = mlp(in_features*2, out_features, **mlp_kwargs)

    def forward(self, data):
        return self.mlp(torch.cat([data.x, data.x**2], dim=1)), None, None


class Single_MP_step(torch.nn.Module):
    '''
    Forward function returns tuple (y_pred, None, None):
    - y_pred: is a node-wise output (e.g. node velocity)
    - The two `None`s are place holders to make model compatible with
      training function in `train.py`.
    '''
    def __init__(self, node_in_features=10, node_out_features=2,
                 edge_in_features=2, message_out_features=5,
                 message_hidden_dims=[10], update_hidden_dims=[],
                 aggr='mean', **mlp_kwargs):
        '''
        Arg-s:
        - node_in_features : #input node features
        - node_out_features: #output node features
        - edge_in_features : #input edge features
        - message_out_features : #message features (edge-wise messages, can be
                              considered as new or intermediate edge features)
        - message_hidden_dims : list of #dims for message MLP=phi. For
                                edge s->t: m_st = phi([x_t - x_s, e_st]).
        - update_hidden_dims : list of #dims for update MLP=gamma. For
                               node i : x_i' = gamma(x_i, Aggregate(m_si))
        - Optional kwargs for both MLPs:
                   defaults are `dropout_p = 0`, `Fn = ReLU`, `Fn_kwargs = {}`.
        '''
        super(Single_MP_step, self).__init__()

        self.message = DiffMessage(node_in_features+edge_in_features,
                                   message_out_features,
                                   hidden_dims=message_hidden_dims,
                                   **mlp_kwargs)
        self.relu = torch.nn.ReLU()

        n_input_features = node_in_features + message_out_features
        self.aggr_update = AggregateUpdate(n_input_features, node_out_features,
                                           hidden_dims=update_hidden_dims,
                                           aggr=aggr, **mlp_kwargs)

    def forward(self, data):
        # convert to undirected graph : cat([e_ij, e_ji])
        edge_index = torch.cat([data.edge_index,
                                torch.stack([data.edge_index[1],
                                             data.edge_index[0]], dim=0)],
                               dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0).contiguous()

        # message
        src, tgt = data.x[edge_index[0]], data.x[edge_index[1]]  # src, tgt features
        m_ij = self.relu(self.message(src, tgt, edge_attr))

        # aggregate and update stages; last layer is linear, i.e. no ReLU()
        x_out = self.aggr_update(data.x, edge_index, m_ij)
        return x_out, None, None


class SingleMP_Tension(torch.nn.Module):
    '''
    Forward function returns tuple (y_pred, tension, None):
    - y_pred: is a node-wise output (e.g. node velocity).
    - tension : edge tensions (for undirected edges).
    - `None` is a place holder to make model compatible with training function in `train.py`.
    '''
    def __init__(self, node_in_features=10, node_out_features=2, edge_in_features=2,
                 message_out_features=5, message_hidden_dims=[10], update_hidden_dims=[],
                 aggr='mean', tension_out_features=1, tension_hidden_dims=[5], **mlp_kwargs):
        '''
        Arg-s:
        - node_in_features : #input node features
        - node_out_features: #output node features
        - edge_in_features : #input edge features
        - message_out_features : #message features (edge-wise messages, can be considered
                                 as new or intermediate edge features )
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
        edge_index = torch.cat([data.edge_index,
                                torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)],
                               dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0).contiguous()

        # message
        src, tgt = data.x[edge_index[0]], data.x[edge_index[1]]  # src, tgt features
        m_ij = self.relu(self.message(src, tgt, edge_attr))

        # aggregate and update stages
        x_out = self.aggr_update(data.x, edge_index, m_ij)  # leave last layer as linear, i.e. no ReLU()

        # tension model
        e_out = self.tension_mlp(m_ij[:m_ij.size(0)//2, :] + m_ij[(m_ij.size(0)//2):, :])

        return x_out, e_out.reshape((e_out.size(0),)), None
