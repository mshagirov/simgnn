import torch
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_scatter import scatter

class mlp(torch.nn.Module):
    '''
    MLP consisting of multiple linear layers w/ activation func-s (`Fn`). Last layer is always linear layer w/o activation.
    '''
    
    def __init__(self, in_features, out_features, hidden_dims=[], dropout_p=0, Fn=ReLU, Fn_kwargs={}):
        '''
        - in_features : input dim-s.
        - out_features: output dim-s. 
        - hidden_dims : a list of hidden dim-s (number of hidden layer neurons) {default : [] an empty list, i.e. no hidden layers}
        - dropout_p   : dropout prob-y for hidden layer(s) {default :  0}.
        - Fn : activation function for hidden layers { default: ReLU }
        - Fn_kwargs : keyword arg-s for `Fn` {default : an empty dict.}
        
        NOTE: The last layer is always linear, i.e. it has no dropout and activation.
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

