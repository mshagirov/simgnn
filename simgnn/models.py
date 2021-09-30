import torch
from torch.nn import ReLU, ModuleList
from simgnn.nn import mlp, SelectiveActivation, IndependentBlock, MessageBlock  # , Encode_Process_Decode
from simgnn.nn import DiffMessage, DiffMessageSquared, AggregateUpdate


class GraphEncoder(torch.nn.Module):
    '''
    Graph encoder model for encoding pt-geometric graph data. Before passing the
    input graph to MLPs, copy of reversed edges (e_ji) is concatenated to the graph
    since datasets in `simgnn.datasets` contain only one copy (e_ij).

    new_edge_index = concatenate(e_ij, e_ji)
    new_edge_attr = concatenate(edge_attr, -edge_attr)  # this passed to MLP

    `GraphEncoder.forward`:
    -----
    Input arg-s:
        - data : pt-geometric graph w/ properties `x`, `edge_attr` and `edge_index`.

    Returns:
        - h_x, edge_index, h_e: processed node, edge variables and concatenated
                                `edge_index`.
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[], **mlp_kwargs):
        '''
        Same input arg-s as the `IndependentBlock` with mode set to `fwd_mode='update'`.

        Uses ReLU as a default activation, to set a different activation function pass keyword arg-s:
            - Fn : default torch.nn.ReLU
            - Fn_kwargs : `dict` of keyword arg-s for constructing `Fn`, default {}.
        Note that same `Fn` is used for both hidden layers of `IndependentBlock` MLPs, and
        for the last layer of `GraphEncoder`.

        Example:
            enc = GraphEncoder(10, 256, Fn=torch.nn.LeakyReLU, Fn_kwargs={'negative_slope':0.2})
            hx, edge_index, he = enc(data)
        '''
        super(GraphEncoder, self).__init__()
        Fn_kwargs = mlp_kwargs['Fn_kwargs'] if 'Fn_kwargs' in mlp_kwargs else {}
        Fn = mlp_kwargs['Fn'] if 'Fn' in mlp_kwargs else ReLU

        self.independent = IndependentBlock(in_dims, out_dims,
                                            hidden_dims=hidden_dims,
                                            fwd_mode='update', **mlp_kwargs)
        # apply activation function on 1st and 3rd input var-s
        self.Fn = SelectiveActivation(var_id=[0, 2], Fn=Fn, Fn_kwargs=Fn_kwargs)

    def forward(self, data):
        # convert to undirected graph : cat([e_ij, e_ji])
        edge_index = torch.cat([data.edge_index,
                                torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)],
                               dim=1).contiguous()
        # edge features for undirected graph : e_ij = - e_ji
        edge_attr = torch.cat([data.edge_attr, -data.edge_attr], dim=0).contiguous()

        return self.Fn(*self.independent(data.x, edge_index, edge_attr))


class GraphDecoder(torch.nn.Module):
    '''
    Graph decoder model for independently processing node and edge variables into velocities and
    tensions.

    `GraphDecoder.forward`:
    -----
    Input arg-s:
        -  x, edge_index, edge_attr : `x` and `edge_attr` are processed using independent MLPs.
                                      `edge_index` is ignored. Before passing it to its MLP,
                                      two halves of `edge_attr` along `axis=0` are summed to pool
                                      edges with opposite directions (new_e = e_ij + e_ji), this
                                      assumes that edges with opposite directions are concatenated
                                      as (i.e. `edge_attr = [e_ij, e_ij]`, same applies to the
                                      `edge_index`, e.g. in the previous message passing layers).

    Returns:
        - y_pred: is a node-wise output (e.g. node velocity).
        - tension : edge tensions (for undirected edges).
        - `None` is a place holder to make model compatible with training function in `train.py`.
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[], **mlp_kwargs):
        super(GraphDecoder, self).__init__()

        self.independent = IndependentBlock(in_dims, out_dims,
                                            hidden_dims=hidden_dims,
                                            fwd_mode='update', **mlp_kwargs)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr[:edge_attr.size(0)//2, :] + edge_attr[(edge_attr.size(0)//2):, :]
        y, _, e_out = self.independent(x, edge_index, edge_attr)
        return y, e_out.reshape((e_out.size(0),)), None


class GraphProcessor(torch.nn.Module):
    '''
    Graph processor model.

    Processes input graph var-s with graph neural network blocks. Blocks repeat
    `n_blocks` times, and each block is followed by same activation function `Fn`
    (default: ReLU). Optionally, can switch between independent and message passing
    layers by setting `block_type` (for all layers), and can disable/enable
    residual or skip connections `x_new = x_old + Fn(layer(x_old))` by setting
    `isresidual`.

    `GraphProcessor.forward`:
    -----
    Input arg-s:
         - x, edge_index, edge_attr : graph var-s

    Returns:
        - x, edge_index, edge_attr : processed graph var-s
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[],
                 aggr='mean', seq='e', block_type='message',
                 n_blocks=5, isresidual=True, **mlp_kwargs):
        '''
        Arg-s:
             - in_dims, out_dims,
               hidden_dims, aggr, seq : agr-s for MessageBlock/IndependentBlock
                                                 (`aggr` and `seq` ignored for
                                                 IndependentBlock)
            - block_type : processor layer types, one of ['message', 'independent']
            - n_blocks   : number of repeating blocks.
            - isresidual : enable/disable residual connections (bool).
        '''
        super(GraphProcessor, self).__init__()

        self.aggr, self.seq = aggr, seq
        self.in_dims, self.out_dims = in_dims, out_dims
        self.hidden_dims, self.mlp_kwargs = hidden_dims, mlp_kwargs

        Fn_kwargs = mlp_kwargs['Fn_kwargs'] if 'Fn_kwargs' in mlp_kwargs else {}
        Fn = mlp_kwargs['Fn'] if 'Fn' in mlp_kwargs else ReLU

        gnn_block = self.get_message if block_type == 'message' else self.get_independent

        self.layers = ModuleList([gnn_block() for k in range(n_blocks)])

        self.Fn = SelectiveActivation(var_id=[0, 2], Fn=Fn, Fn_kwargs=Fn_kwargs)

        self.forward_fn = self.res_fwd if isresidual else self.non_res_fwd

    def get_independent(self):
        return IndependentBlock(self.in_dims, self.out_dims, hidden_dims=self.hidden_dims,
                                fwd_mode='update', **self.mlp_kwargs)

    def get_message(self):
        return MessageBlock(self.in_dims, self.out_dims, hidden_dims=self.hidden_dims,
                            aggr=self.aggr, seq=self.seq, **self.mlp_kwargs)

    def res_fwd(self, x, edge_index, edge_attr):
        for layer in self.layers:
            hx, _, he = self.Fn(*layer(x, edge_index, edge_attr))
            x = (x + hx)/2
            edge_attr = (edge_attr + he)/2
        return x, edge_index, edge_attr

    def non_res_fwd(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x, _, edge_attr = self.Fn(*layer(x, edge_index, edge_attr))
        return x, edge_index, edge_attr

    def forward(self, *xs):
        return self.forward_fn(*xs)


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
