# Model Comparison Results

### formation_energy_per_atom

| Model | MAE | RMSE | R-squared |
|-------|-----|------|-----------|
| RF | 0.0746 | 0.1393 | 0.9810 |
| CGCNN | **0.0341** | **0.0697** | **0.9952** |
| M3GNet† | 0.3210 | 0.4096 | 0.8358 |
| TensorNet | 5.5991 | 7.5781 | -55.2231 |

_† Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

### voltage

| Model | MAE | RMSE | R-squared |
|-------|-----|------|-----------|
| RF | **0.4514** | **0.6380** | **0.6529** |
| CGCNN | 0.4921 | 0.7280 | 0.5482 |
| M3GNet† | 6.2511 | 6.3554 | -33.4367 |
| TensorNet | 34.4206 | 39.2295 | -1311.0687 |

_† Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

### capacity

| Model | MAE | RMSE | R-squared |
|-------|-----|------|-----------|
| RF | 50.2189 | 68.2151 | 0.4302 |
| CGCNN | **48.7821** | **66.0855** | **0.4652** |
| M3GNet† | 162.6733 | 186.0789 | -3.2398 |
| TensorNet | 169.8472 | 192.2349 | -3.5250 |

_† Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

### energy_above_hull

| Model | MAE | RMSE | R-squared |
|-------|-----|------|-----------|
| RF | 0.0278 | 0.0683 | 0.3826 |
| CGCNN | **0.0211** | **0.0484** | **0.6903** |
| M3GNet† | 3.4989 | 3.6105 | -1724.6533 |
| TensorNet | 29.7300 | 42.5254 | -239399.4541 |

_† Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_
