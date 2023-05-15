# Notes
---

## Tasks 👷 🚧 
- Edit nb 4: positional encoding for tension GNN
    - [ ] implement x_e w/ posenc
    - [ ] test simple GNN training and pred for T
    - [ ] devise experiments to explore posenc
    - [ ] test removing norms in diffX (no posenc)
    - [ ] posenc with suitable diffX
    - [ ] posenc no diffX or norms (ablation tests)
    
---
## Node-to-Cell Encoding/Pooling Layer 🧱 
> This type GNN layers are not yet used but can be implemented using message passing method as shown below
1. Initiate node-to-cell edge attr-s as (source) node attr-s `x[node2cell_index[0]]`.
1. Compute node-to-cell edge attr-s using MLP: `e_n2c = MLP( x[node2cell_index[0]] )`
1. Aggregate node-to-cell edge attr-s as cell attr-s : `x_cell = Aggregate(e_n2c)`
1. Compute new cell attr-s using (encodes `x_cell` into cell attr-s) : `h_cell = MLP_Cell_encoder( x_cell )`

```python
n2c_model = mlp(...) # "message", just node-wise MLP
cell_aggr = Aggregate()
cell_enc = mlp(...)

e_n2c = n2c_model(data.x)[data.node2cell_index[0]]
x_cell = cell_aggr(data.cell_pressures.size(0), data.node2cell_index, e_n2c)
h_cell = cell_enc(x_cell)
```

---
## Future Work 🔮 
- Model architecture:
    - add *Cell layer* processor
    - compare MLP to CONV layers for message passing.
    - dynamic graphs based on relative positions, and use cell edges and cell attrib only for queries on `Y_edge`, `Y_cell`.
- Noise and Loss
    - rollout vs 1-step losses, and multi-step loss.
    - Brownian noise for velocity (e.g., Sanchez-Gonzalez, *et al.* \[ASG2020\])