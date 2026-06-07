import h5py
import numpy as np
import torch
import torch.nn.functional as F


def load_last_flat_samples(file_path, dataset, n_samples):
    with h5py.File(file_path, 'r') as f:
        dset = f[dataset]
        n_case, n_time = dset.shape[:2]
        total = n_case * n_time
        start = max(0, total - n_samples)
        pieces = []
        while start < total:
            case_id = start // n_time
            t0 = start % n_time
            take = min(n_time - t0, total - start)
            pieces.append(dset[case_id, t0:t0 + take])
            start += take
    return np.concatenate(pieces, axis=0).astype(np.float32)


def grid(size_x, size_y, device):
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
    gridx = gridx.reshape(size_x, 1, 1).repeat([1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
    gridy = gridy.reshape(1, size_y, 1).repeat([size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)


@torch.no_grad()
def eval_trunk_chunked(net, size_x, size_y, latent_dim, device, chunk_size=65536):
    if net.__class__.__name__ == 'FNO2':
        out = net(grid(size_x, size_y, device)).detach().cpu()
        return out.permute(2, 0, 1).contiguous().numpy()

    g = grid(size_x, size_y, device).reshape(-1, 2)
    out = torch.empty(g.shape[0], latent_dim, dtype=torch.float32, device='cpu')
    for start in range(0, g.shape[0], chunk_size):
        out[start:start + chunk_size] = net(g[start:start + chunk_size]).detach().cpu()
    return out.reshape(size_x, size_y, latent_dim).permute(2, 0, 1).contiguous().numpy()


def _write_or_replace(handle, key, value):
    if key in handle:
        if handle[key].shape == value.shape:
            handle[key][...] = value
        else:
            del handle[key]
            handle.create_dataset(key, data=value, dtype='float32')
    else:
        handle.create_dataset(key, data=value, dtype='float32')


def load_or_create_multiscale_trunk(model, file_path, name_without_ext, device):
    keys = {
        'f': f'trunk_{name_without_ext}_f',
        'm': f'trunk_{name_without_ext}_m',
        'c': f'trunk_{name_without_ext}_c',
    }
    with h5py.File(file_path, 'a') as f:
        if all(k in f for k in keys.values()):
            trunk_f = f[keys['f']][:]
            trunk_m = f[keys['m']][:]
            trunk_c = f[keys['c']][:]
        else:
            x, y = model.input_dim
            trunk_f = eval_trunk_chunked(model.trunk_net_f, x, y, model.latent_dim_f, device)
            trunk_m = eval_trunk_chunked(model.trunk_net_m, x // 2, y // 2, model.latent_dim_m, device)
            trunk_c = eval_trunk_chunked(model.trunk_net_c, x // 4, y // 4, model.latent_dim_c, device)
            _write_or_replace(f, keys['f'], trunk_f.astype(np.float32))
            _write_or_replace(f, keys['m'], trunk_m.astype(np.float32))
            _write_or_replace(f, keys['c'], trunk_c.astype(np.float32))
    return (
        torch.from_numpy(trunk_f).to(device),
        torch.from_numpy(trunk_m).to(device),
        torch.from_numpy(trunk_c).to(device),
    )


def decode_multiscale_from_trunk(model, z, trunk_f, trunk_m, trunk_c):
    zf, zm, zc = torch.split(z, [model.latent_dim_f, model.latent_dim_m, model.latent_dim_c], dim=-1)
    yf = torch.einsum('bi,ixy->bxy', zf, trunk_f)
    ym = torch.einsum('bi,ixy->bxy', zm, trunk_m)
    yc = torch.einsum('bi,ixy->bxy', zc, trunk_c)
    ym = F.interpolate(ym.unsqueeze(1), size=yf.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
    yc = F.interpolate(yc.unsqueeze(1), size=yf.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
    return (yf + ym + yc).unsqueeze(1)


def load_or_create_single_trunk(model, file_path, name_without_ext, device):
    key = f'trunk_{name_without_ext}'
    with h5py.File(file_path, 'a') as f:
        if key in f:
            trunk = f[key][:]
        else:
            x, y = model.input_dim
            trunk = eval_trunk_chunked(model.trunk_net, x, y, model.latent_dim, device)
            _write_or_replace(f, key, trunk.astype(np.float32))
    return torch.from_numpy(trunk).to(device)


def decode_single_from_trunk(model, z, trunk):
    if z.ndim == 2 and z.shape[-1] == model.latent_dim:
        z = z.view(z.shape[0], 2, int(model.latent_dim / 2))
    trunk_xy = trunk.permute(1, 2, 0)
    trunk_xy = trunk_xy.view(trunk_xy.shape[0], trunk_xy.shape[1], 2, int(model.latent_dim / 2))
    return torch.einsum('bci,xyci->bcxy', z, trunk_xy)
