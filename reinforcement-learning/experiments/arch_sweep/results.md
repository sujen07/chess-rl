| Run | Model | Params | Val acc | Δ acc | Val policy loss | Val value loss | Train pos/s | Infer b=1 (ms) | Infer b=64 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | resnet 64ch x6 | 1.04M | 43.92% | +0.00 pp | 1.8185 | 0.2581 | 22,705 | 1.23 | 1.35 |
| baseline_seed1 | resnet 64ch x6 | 1.04M | 43.68% | -0.24 pp | 1.8205 | 0.2577 | 22,622 | 1.26 | 1.36 |
| se | resnet 64ch x6 +SE | 1.05M | 44.27% | +0.35 pp | 1.8052 | 0.2581 | 19,550 | 2.04 | 2.19 |
| transformer | transformer d128 x4L | 1.09M | 38.28% | -5.64 pp | 2.0726 | 0.2549 | 18,191 | 1.25 | 2.39 |
| transformer_bias | transformer d128 x4L +attn bias | 1.16M | 39.07% | -4.84 pp | 2.0373 | 0.2537 | 17,524 | 1.29 | 2.43 |
| hybrid | resnet 64ch x4 +SE +2 attn | 0.97M | 43.80% | -0.12 pp | 1.8233 | 0.2577 | 18,287 | 2.09 | 2.18 |
