import torch
import torch.nn as nn


if __name__ == '__main__':
    v = torch.arange(100)
    v1 = v.view(20, 5)
    print(v1[:, 0])

    v2 = v.view(5, 20)
    print(v2[0, :])