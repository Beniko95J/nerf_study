import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max())
    mi = float(x.min())
    d = ma - mi if ma != mi else 1e5

    return (x - mi) / d


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)

    return ListedColormap(high_res)


COLORMAPS = {'magma': high_res_colormap(cm.get_cmap('magma')),
             'viridis': high_res_colormap(cm.get_cmap('viridis')),
             'plasma': high_res_colormap(cm.get_cmap('plasma')),
             'gray': high_res_colormap(cm.get_cmap('gray')),
             'jet': high_res_colormap(cm.get_cmap('jet'))}


def tensor2array(tensor, max_value=None, colormap='jet'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()

    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy() / (max_value + 1e-10)
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225

    return array


def save_disp_vis(filename, disp):
    disp = normalize_image(disp)
    disp = COLORMAPS['jet'](disp).astype(np.float32)[..., :3]
    plot_data = np.zeros((disp.shape[0], disp.shape[1], 3))
    plot_data[:, :] = disp

    _, ax = plt.subplots()
    ax.imshow(plot_data)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
