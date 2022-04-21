import torch
from torch.nn import ReLU, ModuleList, Dropout, BatchNorm1d, LayerNorm
from torch_scatter import scatter
from simgnn.nn import dims_to_dict, mlp, SelectiveLayer, Residual, SequentialUpdate
from simgnn.nn import IndependentBlock, MessageBlock, Encode_Process_Decode_ptgraph
from simgnn.nn import DiffMessage, DiffMessageSquared, AggregateUpdate

from collections import OrderedDict


def get_simple_gnn(n_blocks=3, dropout_p=0.1, device=torch.device('cpu'), is_residual=True,
                  input_dims=None,  output_dims=None):
    '''
    latent_layer_n
    '''    
    # dropout for hidden layers & GN blocks (if any)
    latent_dims = 128
    latent_layer_n = 2
    
    if input_dims==None:
        input_dims = OrderedDict([('node', 10), ('edge', 2)]) # node_features, edge_features
    if output_dims==None:
        output_dims = OrderedDict([('node', 2), ('edge', 1)]) # velocity:(Nv,2), tensions:(Ne,1)
    

    encoder_kwrgs    = {'hidden_dims':[128, 128],
                        'dropout_p': dropout_p, # dropout for hidden layers (if any)
                       }

    processor_kwargs = {'n_blocks': n_blocks,
                        'block_type': 'message',
                        'is_residual': is_residual,
                        'seq': 'n', # 'n', 'e', 'p'
                        'norm_type': 'ln', # 'ln' or 'bn'
                        'block_p': dropout_p,  # block dropout (last layer)
                        'dropout_p': dropout_p, # dropout for hidden layers (if any)
                        'hidden_dims':[latent_dims for n in range(latent_layer_n)]
                       }

    decoder_kwargs   = {'hidden_dims':[128, 128, 16],
                        'dropout_p': dropout_p, # dropout for hidden layers (if any)
                       }

    net = construct_simple_gnn(input_dims, latent_dims, output_dims,
                               encoder_kwrgs=encoder_kwrgs,
                               processor_kwargs=processor_kwargs, 
                               decoder_kwargs=decoder_kwargs).to(device)
    return net


def construct_simple_gnn(input_dims, latent_dims, output_dims,
                         encoder_kwrgs={}, processor_kwargs={}, decoder_kwargs={}):
    return Encode_Process_Decode_ptgraph(GraphEncoder(input_dims, latent_dims, **encoder_kwrgs),
                                         GraphProcessor(latent_dims, latent_dims, **processor_kwargs),
                                         GraphDecoder(latent_dims, output_dims, **decoder_kwargs))


class GraphEncoder(torch.nn.Module):
    '''
    Graph encoder model for encoding pt-geometric graph data. Assumes that a copy of reversed edges (e_ji)
    with their corresponding edge features are concatenated to the graph.

    - Note that datasets in `simgnn.datasets` contain only one copy of edges (e_ij).

    `GraphEncoder.forward`:
    -----
    Input arg-s:
        - data : pt-geometric graph w/ properties `x`, `edge_attr` and `edge_index` (undirected graph).

    Returns:
        - data with two new variables `h_v` (node features) and `h_e` (edge features).
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[], block_Fn=ReLU, block_Fn_kwargs={}, **mlp_kwargs):
        '''
        Same input arg-s as the `IndependentBlock` with mode set to `fwd_mode='update'`.

        Additional keyword arg-s:
        - block_Fn : activation function for the block.
        - block_Fn_kwargs :  dict of keyword arg-s for `block_Fn`.

        To set activation functions of MLPs use `Fn` and `Fn_kwargs`.

        Example:
            enc = GraphEncoder(10, 256, block_Fn=torch.nn.LeakyReLU, Fn_kwargs={'negative_slope':0.2})
            data = enc(data) # adds new variables data.h_v, data.h_e
        '''
        super(GraphEncoder, self).__init__()

        self.independent = IndependentBlock(in_dims, out_dims,
                                            hidden_dims=hidden_dims,
                                            fwd_mode='update', **mlp_kwargs)
        # apply activation function on 1st and 3rd input var-s
        self.Fn = SelectiveLayer(block_Fn(**block_Fn_kwargs))

    def forward(self, d):
        d.h_v, _, d.h_e = self.Fn(*self.independent(d.x, d.edge_index, d.edge_attr))
        return d


class GraphDecoder(torch.nn.Module):
    '''
    Graph decoder model for independently processing node and edge variables into velocities and
    tensions.

    `GraphDecoder.forward`:
    -----
    Input arg-s:
        -  data : pt-geometric graph w/ `h_v` (node) and `h_e` (edge) features (processed using independent MLPs),
                  `edge_index` is ignored. Before passing it to its MLP, `h_e` are pooled along `dim=0` using
                  `torch_scatter.scatter()` and `data.edge_id`. Set pooling/aggregation scheme w/ `edge_reduce`
                  keyword argument.
    Returns:
        - y_pred: is a node-wise output (e.g. node velocity).
        - tension : edge tensions (for undirected edges).
        - `None` is a placeholder to make model compatible with training function in `train.py`.
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[], edge_reduce='sum', **mlp_kwargs):
        super(GraphDecoder, self).__init__()
        self.edge_reduce = edge_reduce
        self.independent = IndependentBlock(in_dims, out_dims,
                                            hidden_dims=hidden_dims,
                                            fwd_mode='update', **mlp_kwargs)

    def forward(self, d):
        d.h_e = scatter(d.h_e, d.edge_id, dim=0, reduce=self.edge_reduce, dim_size=d.num_edges//2)
        # "update" mode: passes d.edge_index unchanged and ignored
        d.h_v, _, d.h_e = self.independent(d.h_v, d.edge_index, d.h_e)
        return d.h_v, d.h_e.reshape((d.h_e.size(0),)), None


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
         - data : pt-geometric graph w/ fields `edge_index`,  `h_v` (node features) and `h_e` (edge features).

    Returns:
        - data : processed graph, with new the `h_v` and `h_e` var-s.
    '''
    def __init__(self, in_dims, out_dims, hidden_dims=[],
                 aggr='mean', seq='e', block_type='message',
                 n_blocks=1, is_residual=True, block_Fn=ReLU,
                 block_Fn_kwargs={}, block_p=0, block_norm=True,
                 norm_type='ln', **mlp_kwargs):
        '''
        Arg-s:
             - in_dims, out_dims,
               hidden_dims, aggr, seq : agr-s for MessageBlock/IndependentBlock
                                                 (`aggr` and `seq` ignored for
                                                 IndependentBlock)
            - block_type : processor layer types, one of ['message', 'independent']
            - n_blocks   : number of repeating blocks.
            - is_residual : enable/disable residual connections (bool).
            - block_Fn : activation function for the block.
            - block_Fn_kwargs :  dict of (same) keyword arg-s for `block_Fn`.
            - block_p : block dropout prob-y. Uses same prob-s for both nodes and
                        edges if `block_p` is `float`, or individual "node" and "edge"
                        dropout prob-s if `block_p` is a dict w/ keys "node" and "edge".
            - block_norm : enable/disable normalisation layer for the block.
            - norm_type : one of 'ln' (LayerNorm) or 'bn' (BatchNorm1d).
        '''
        super(GraphProcessor, self).__init__()

        self.aggr, self.seq = aggr, seq
        self.mlp_kwargs = mlp_kwargs

        block_p, self.in_dims, self.out_dims, self.hidden_dims = dims_to_dict(block_p,
                                                                              in_dims,
                                                                              out_dims,
                                                                              hidden_dims)
        # GraphNet block constructor
        if block_type == 'message':
            get_block = self.get_message
        elif block_type == 'independent':
            get_block = self.get_independent

        # Normalisation layer constructor
        norm_layer = LayerNorm if norm_type == 'ln' else BatchNorm1d

        gnn_layers = []
        for k in range(n_blocks):
            block = [get_block(), SelectiveLayer(block_Fn(**block_Fn_kwargs))]

            # normalisation layers
            if block_norm:
                block.append(SelectiveLayer(norm_layer(self.out_dims['node']), var_id=0))
                block.append(SelectiveLayer(norm_layer(self.out_dims['edge']), var_id=2))

            # Dropout layers
            if block_p['node'] > 0:
                block.append(SelectiveLayer(Dropout(block_p['node']), var_id=0))
            if block_p['edge'] > 0:
                block.append(SelectiveLayer(Dropout(block_p['edge']), var_id=2))

            if is_residual:
                gnn = Residual(SequentialUpdate(*block))
            else:
                gnn = SequentialUpdate(*block)
            gnn_layers.append(gnn)

        self.layers = ModuleList(gnn_layers)

    def get_independent(self):
        gnn = IndependentBlock(self.in_dims, self.out_dims, hidden_dims=self.hidden_dims,
                               fwd_mode='update', **self.mlp_kwargs)
        return gnn

    def get_message(self):
        gnn = MessageBlock(self.in_dims, self.out_dims, hidden_dims=self.hidden_dims,
                           aggr=self.aggr, seq=self.seq, **self.mlp_kwargs)
        return gnn

    def forward(self, d):
        for layer in self.layers:
            d.h_v, _, d.h_e = layer(d.h_v, d.edge_index, d.h_e)
        return d


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
