def Name_AE(cfg):
    if cfg.pattern == 'vani':
        name = f'{cfg.model}_d{cfg.latent_dim}'
    else:
        name = f'{cfg.model}_d{cfg.latent_dim}_{cfg.pattern}'
    return name

