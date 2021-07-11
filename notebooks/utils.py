import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def viz_tensors(tensor, nrow=4, normalize=False, mean=None, std=None):
    # if std:
    #     std = torch.tensor(std)[:, None, None]
    #     tensor = tensor * std

    # if mean:
    #     mean = torch.tensor(mean)[:, None, None]
    #     tensor = tensor + mean

    grid = make_grid(tensor[: nrow * nrow], nrow=nrow, normalize=normalize)
    plt.imshow(grid.permute(1, 2, 0))

    return grid
