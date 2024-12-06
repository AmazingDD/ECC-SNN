# ECC-SNN
Official code for "ECC-SNN: " (XXX2025)

## How to run

### Setup Stage: run `prepare.py`

- overlapping setup example
    ```
    python prepare.py -dataset=cifar100 -nt=5 -b=32 -edge=svgg -cloud=vgg -ce=150 -ee=200 -distill
    ```

- non-overlapping setup example
    ```
    python prepare.py -dataset=cifar100 -nt=5 -b=32 -edge=svgg -cloud=resnet50 -ce=30 -ee=200  -pretrain -distill
    ```

### Running Stage: run `update.py` and `execution.py`

- preparing models for each task
    ```
    python update.py
    ```
- evaluating performance
    ```
    python execution.py
    ```