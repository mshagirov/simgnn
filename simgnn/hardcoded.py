from collections import OrderedDict
from simgnn.models import construct_simple_gnn

def get_model_04122021():
    '''
    input_dims = OrderedDict([('node', 10), ('edge', 18)]) # node_features, edge_features
    output_dims = OrderedDict([('node', 2), ('edge', 1)]) # velocity:(Nv,2), tensions:(Ne,1)
    latent_dims = 128

    encoder_kwrgs    = {'hidden_dims':[64]}
    processor_kwargs = {'n_blocks': 10,
                        'block_type': 'message',
                        'is_residual': True,
                        'seq': 'n', # 'n', 'e', 'p'
                        'norm_type': 'bn', # 'ln' or 'bn'
                        'block_p': 0,  # block dropout (last layer)
                        'dropout_p': 0, # dropout for hidden layers (if any)
                        'hidden_dims':[]
                       }
    decoder_kwargs   = {'hidden_dims':[64, 32, 8]}
    '''
    input_dims = OrderedDict([('node', 10), ('edge', 18)]) # node_features, edge_features
    output_dims = OrderedDict([('node', 2), ('edge', 1)]) # velocity:(Nv,2), tensions:(Ne,1)
    latent_dims = 128

    encoder_kwrgs    = {'hidden_dims':[64]}
    processor_kwargs = {'n_blocks': 10,
                        'block_type': 'message',
                        'is_residual': True,
                        'seq': 'n', # 'n', 'e', 'p'
                        'norm_type': 'bn', # 'ln' or 'bn'
                        'block_p': 0,  # block dropout (last layer)
                        'dropout_p': 0, # dropout for hidden layers (if any)
                        'hidden_dims':[]
                       }
    decoder_kwargs   = {'hidden_dims':[64, 32, 8]}

    net = construct_simple_gnn(input_dims, latent_dims, output_dims,
                               encoder_kwrgs=encoder_kwrgs,
                               processor_kwargs=processor_kwargs, 
                               decoder_kwargs=decoder_kwargs)
    return net
