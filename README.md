## `SyncBrain`: Exploring Brain Functional Dynamics Through Neural Oscillatory Synchronization

```plaintext
AAAI26/
 ├─ src/
 │   ├─ data/
 │   │   ├─ create_dataset.py    
 │   │   └─ dataset.py          # Loading brain data (HCP-A, HCP-YA, HCP-WM, ADNI, OASIS, PPMI, NIFD)
 │   ├─ modules/
 │   │   ├─ GST.py              # Graph Sattering Transform
 │   │   └─ SyncBrain_solver.py    
 │   ├─ SyncBrain.py            # The main SyncBrain model
 │   └─ utils.py                  
 ├─ train_and_eval.py  
