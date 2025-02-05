# Notes
---

## Updates ‼️
### Issues due to addition of kwargs and args in PyG v2.6.1

Test batching with Python's next and iter func-s.

```python
batch = next(iter(loaders['val']))
batch
```

Changes on `simgnn.datasets.CellData` to fix the issue.

```shell
diff --git a/simgnn/datasets.py b/simgnn/datasets.py
index b30d127..f48d5aa 100644
--- a/simgnn/datasets.py
+++ b/simgnn/datasets.py
@@ -73,7 +73,7 @@ class CellData(Data):
     def num_cells(self, val):
         self.__num_cells__ = val
 
-    def __inc__(self, key, value):
+    def __inc__(self, key, value,*args,**kwargs):
         if key == 'node2cell_index':
             return torch.tensor([[self.num_nodes], [self.num_cells]])
         if key == 'cell2node_index':
@@ -81,13 +81,13 @@ class CellData(Data):
         if key == 'edge_id':
             return torch.unique(self.edge_id).size(0)
         else:
-            return super(CellData, self).__inc__(key, value)
+            return super(CellData, self).__inc__(key, value,*args,**kwargs)
 
-    def __cat_dim__(self, key, value):
+    def __cat_dim__(self, key, value,*args,**kwargs):
         if key == 'node2cell_index' or key == 'cell2node_index':
             return -1
         else:
-            return super(CellData, self).__cat_dim__(key, value)
+            return super(CellData, self).__cat_dim__(key, value,*args,**kwargs)
```

## Tasks 👷 🚧 
- Edit nb 4: positional encoding for tension GNN
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