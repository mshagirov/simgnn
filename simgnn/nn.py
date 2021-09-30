import torch
from torch.nn import ModuleDict, ModuleList, Sequential, Linear, ReLU, Dropout
from torch_scatter import scatter
from collections import OrderedDict


def dims_to_dict(*mlp_dims):
    '''
    Converts/broadcasts MLP dimension arg-s `mlp_dims` (int, dict, OrderedDict)
    to a set of OrderedDict's with same keys (keys represent graph variables).
    Keys of the dict/OrderedDict input arg-s in `mlp_dims` are used if any of
    the input arg-s is a dict or OrderedDict, or default ("node", "edge") keys
    are used otherwise. Input arg-s must have the same keys (and same ordering)
    if more than one of the `mlp_dims` are dict/OrderedDict's.

    For python versions before v3.7 use OrderedDict in order to maintain correct
    order of the dictionary keys.

    Examples:
        c_i, c_o, c_h, ... = dims_to_dict(10, 128, 64, ...)
        # use dict/OrderedDict inputs to specify var dim-s:
        c_i, c_o, c_h = dims_to_dict(8, 16, {'a':10,'b':2, 'c': 3})
    '''
    def is_dict(d):
        return True if type(d) == dict or type(d) == OrderedDict else False

    n_vars = max([(k, len(mlp_dim) if is_dict(mlp_dim) else 0)
                  for k, mlp_dim in enumerate(mlp_dims)], key=lambda x: x[1])

    var_names = ("node", "edge") if n_vars[1] == 0 \
        else tuple(mlp_dims[n_vars[0]].keys())

    mlp_dims_out = (
        OrderedDict(
                    ((var_k, mlp_dim[var_k] if is_dict(mlp_dim) else mlp_dim)
                        for var_k in var_names)) for mlp_dim in mlp_dims
                    )
    mlp_dims_out = tuple(mlp_dims_out)
    mlp_dims_out = mlp_dims_out if len(mlp_dims_out) > 1 else mlp_dims_out[0]
    return mlp_dims_out


class Encode_Process_Decode(torch.nn.Module):
    '''
    Encode-Process-Decode framework from "Learning to Simulate Complex Physics with Graph Networks" by
    A. Sanchez-Gonzalez et al. (2020)
    '''
    def __init__(self, encoder, processor, decoder):
        '''
        Arg-s:
            encoder, processor, decoder : torch.nn.Module objects

            - encoder : accepts variable number of arg-s equal to the num. of arg-s for the forward
                        function of `Encode_Process_Decode`
            - processor : outputs from the `encoder` are fed to the processor.
            - decoder : accepts outputs from the `processor`.

        Returns `output` from:
            output = decoder(*processor(*encoder(*X)))
        '''
        super(Encode_Process_Decode, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def forward(self, *X):
        return self.decoder(*self.processor(*self.encoder(*X)))


class SelectiveActivation(torch.nn.Module):
    '''
    Apply given activation function Fn on var_id`th input variable(s).

    Arg-s:
        - var_id : index of input variable, can be an integer or a list
                   of integers.
        - Fn : activation function.
        - Fn_kwargs : dict of keyword arg-s for construcing `Fn`.

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


class mlp(torch.nn.Module):
    '''
    MLP consisting of multiple linear layers w/ activation func-s (`Fn`). Last
    layer is always linear layer w/o activation or dropout.
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

    Notes:
        - `DiffMessage` is a redundant class. All of its functionality can be
          reproduced using `Message` class.
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
    Aggregates edge features (`edge_attr`) from neighbouring nodes and updates
    node features.
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
        - in_features : input dim-s == 2*`#src_features` + `#edge_features`.
                        Assumes `#tgt_features`==`#src_features`.
        - out_features: output dim-s, e.g. `#edge_features`.

        Optional kwargs for `mlp`:
                hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(DiffMessageSquared, self).__init__()
        self.mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, src, tgt, edge_attr):
        '''
        - src, tgt : source and target features w/ shapes (#edges, #src_features)
                     and (#edges, #tgt_features)
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
    graph variable. The sequence of aggregate+update steps for variables {node,
    edge} can be one of {"e": edge-then-node, "n": node-then-edge, or
    "p": parallel/simulataneous} update schemes given by `seq` argument.

    Notes:
        - Aggregation of nodes to edges is fixed to concatenation operation, whereas
        aggregation of edges to nodes can be one of the schemes set by an input
        argument `aggr`.
        - Update function can be a linear layer or a MLP. For all cases, the last
        layer is always linear.
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[],
                 aggr='mean', seq='e', **mlp_kwargs):
        '''
        Arg-s:
            - in_dims, out_dims : number of input and output dimensions. Either an
                      int (if all MLPs have same input/output dim-s) or a dict of
                      integers w/ keys "node" and "edge",e.g.{'node':10,'edge':2,...}.
            - hidden_dims : a list of hidden dimensions {default : [] no hidden
                      layers}, or a dict of lists,e.g.{'node':[],'edge':[8,16],...}.
            - aggr : message aggregation scheme, one of `['sum', 'mul', 'mean',
                     'min', 'max']` {default: 'mean'}.
            - seq : update sequence, one of 'e' (edge-then-node), 'n' (node-then-edge),
                    or 'p' (parallel or simulataneous update).
            - mlp_kwargs : optional kwarg-s for MLPs.

        Notes:
            - Dimensions of inputs for actual node and edge MLPs depend on update
              sequence `seq`.
            - In order to have same output or hidden dimensions for all MLPs use
              integers for `out_dims`, and a list for `hidden_dims`, instead of
              dictionaries.
        '''
        super(MessageBlock, self).__init__()
        assert seq in 'enp'

        in_dims, out_dims, hidden_dims = dims_to_dict(in_dims, out_dims, hidden_dims)

        if seq == 'e':
            # update edge-then-node {default}
            f_v_in = in_dims['node'] + out_dims['edge']
            f_e_in = in_dims['edge'] + 2*in_dims['node']
            seq = ['edge', 'node']  # update order
        elif seq == 'n':
            # update node-then-edge
            f_v_in = in_dims['node'] + in_dims['edge']
            f_e_in = in_dims['edge'] + 2*out_dims['node']
            seq = ['node', 'edge']  # update order
        elif seq == 'p':
            # parallel update
            f_v_in = in_dims['node'] + in_dims['edge']
            f_e_in = in_dims['edge'] + 2*in_dims['node']

        f_var = {'node': AggregateUpdate(f_v_in, out_dims['node'], hidden_dims=hidden_dims['node'],
                                         aggr=aggr, **mlp_kwargs),
                 'edge': Message(f_e_in, out_dims['edge'], hidden_dims=hidden_dims['edge'],
                                 **mlp_kwargs)}

        if seq == 'p':
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


class IndependentBlock(torch.nn.Module):
    '''
    Layer w/ independent MLPs that all use the same `mlp_kwargs`. The last
    layer of all MLPs are plain linear layers, i.e. they have no dropout/
    activations.

    Examples:
        - Process mode :
            # Inputs --> x1: [#nodes, 10]; x2: [#edges, 2], ...
            # Enc : encodes node and edge features to 32-dim vectors (MLPs w/
            # one 16-dim hidden layer)
            Enc = IndependentBlock({'node':10,'edge':2, ...}, 32, hidden_dims=[16])
            x1_new, x2_new, ... = Enc(x1, x2, ...);

        - Update mode :
            L = IndependentBlock(..., fwd_mode="update")
            # edge_index is unchanged
            x_new, edge_index, e_new = L(x, edge_index, edge_attr)
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[], fwd_mode="process", **mlp_kwargs):
        '''
        Arg-s:
            - in_dims, out_dims : number of input and output dim-s. Either an
                                  int (if all MLPs have same input/output dim-s) or a dict of
                                  integers w/ keys "node", "edge", etc..
                                  E.g. {'node':10,'edge':2, ...}
            - hidden_dims : a list of hidden dimensions {default : [] no hidden
                            layers}, or a dict of lists. E.g. {'node':[],'edge':[8,16], ...}
            - fwd_mode : {Update mode} If `fwd_mode="update"` the forward function
                         accepts three input arg-s `(x, edge_index, edge_attr)`.
                         {Process mode} if `fwd_mode="process"`, the forward accepts
                         same number of inputs as the number of keys in `in_dims`,
                         `out_dims`, and `hidden_dims` (if they are not dict, then
                         it is assumed that there are two keys: "node" and "edge").
            - mlp_kwargs : kwarg-s for MLPs.

        Notes:
            - In order to have same input, output or hidden dimensions for all
              MLPs, use integers for `in_dims`, `out_dims`, and a list for
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
            - The forward function of `IndependentBlock` can have two different
              modes, "update" and "process". In update mode (fwd_mode="update")
              `IndependentBlock` updates only node (x) and edge variables (edge_attr)
              and has a consistent input/output signature with update functions and
              `MessageBlock`. The order of the inputs to the forward function is
              `(x, edge_index, edge_attr)`, where `edge_index` is ignored and only
              node and edge var-s are processed using independent MLPs. In update
              mode, if any of `in_dims`, `out_dims`, or `hidden_dims` are dict or
              OrderedDict, then node and edge keys must be "node" and "edge" respectively.
        '''
        super(IndependentBlock, self).__init__()

        in_dims, out_dims, hidden_dims = dims_to_dict(in_dims, out_dims,
                                                      hidden_dims)

        self.mlp_dict = ModuleDict({k: mlp(in_dims[k], out_dims[k],
                                           hidden_dims=hidden_dims[k],
                                           **mlp_kwargs) for k in in_dims})

        self.forward_fn = self.update_fn if fwd_mode == "update" else self.process_fn

    def update_fn(self, x, edge_index, edge_attr):
        return self.mlp_dict['node'](x), edge_index, self.mlp_dict['edge'](edge_attr)

    def process_fn(self, *xs):
        ys = []
        for x, k in zip(xs, self.mlp_dict):
            ys.append(self.mlp_dict[k](x))
        return tuple(ys)

    def forward(self, *xs):
        return self.forward_fn(*xs)
