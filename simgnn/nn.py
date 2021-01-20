import torch
from torch.nn import Sequential, Linear, ReLU, Dropout

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


class EdgeModel(torch.nn.Module):
    def __init__(self, in_features, out_features, **mlp_kwargs):
        '''
        Arg-s:
        - in_features : input dim-s == `#src_features` + `#tgt_features` + `#edge_features`.
        - out_features: output dim-s, e.g. `#edge_features`. 
        
        Optional kwargs for `mlp`: hidden_dims =[], dropout_p = 0, Fn = ReLU, Fn_kwargs = {}.
        '''
        super(EdgeModel, self).__init__()
        self.edge_mlp = mlp(in_features, out_features, **mlp_kwargs)

    def forward(self, src, tgt, edge_attr):
        '''
        - src, tgt : source and target features w/ shapes (#edges, #src_features) and (#edges, #tgt_features)
        - edge_attr : edge features w/ shape (#edges, #edge_features)
        '''
        return self.edge_mlp( torch.cat( [src, tgt, edge_attr], 1) )

