# Pytorch Implementation of DUCK-Net for polyp image segmentation

Pytorch Implementation of DUCK-Net

Original code is available in [github](https://github.com/RazvanDu/DUCK-Net) and you can see detail of this model in [paper](https://www.nature.com/articles/s41598-023-36940-5).

## Installation

**Requirements**

-   Python >= 3.10
-   [Pytorch](https://pytorch.org/get-started/locally/) >= 2.2.0
-   CUDA 12.0

It's expected to work for latest versions too.

```bash
git clone https://github.com/russel0719/DUCK-Net-Pytorch.git
pip install torch torchvision pillow tqdm
```

## Train

1. Fix [config](./utils/config.py) directly or Fix codes in [train.py](train.py) under `if __name__ == "__main__"` statement.
2. Run train.py

```bash
cd DUCK-Net-Pytorch
python train.py
```

## Predict

1. Fix [config](./utils/config.py) directly and Fix paths in [predict.py](predict.py) under `if __name__ == "__main__"` statement.
2. Run predict.py

```bash
cd DUCK-Net-Pytorch
python predict.py
```

## Reference

-   Paper

    -   [Using DUCK-Net for polyp image segmentation](https://www.nature.com/articles/s41598-023-36940-5)

-   Original Github

    -   [Tensorflow Implementation of RazvanDu](https://github.com/RazvanDu/DUCK-Net)
