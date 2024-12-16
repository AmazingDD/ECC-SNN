# ECC-SNN
Official code for "ECC-SNN: " (XXX2025)

## How to run

### Setup Stage:

```
python prepare.py -nt=5 -patience=50 -b=64 -ce=30 -ee=200 -cloud=vit -edge=svgg -dataset=cifar100 -T=4 -gpu=0 -distill -pretrain
```

### Running Stage:

- preparing models for each task
    ```
    python update.py
    ```
- evaluating performance
    ```
    python execution.py
    ```