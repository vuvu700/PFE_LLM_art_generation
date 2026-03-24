"""
pour ajouter un preset il suffit de rajouter une ligne avec les differents parametres
"""
PRESETS = {
    "1.6M": dict(depth=6, head_dim=128, context_size=4096, nb_heads_mult=1),
    "1.6M_ext": dict(depth=6, head_dim=128, context_size=4096 * 4, nb_heads_mult=1),
    "5.5M": dict(depth=6, head_dim=256, context_size=4096, nb_heads_mult=1),
    "5.5M_ext": dict(depth=6, head_dim=256, context_size=4096 * 4, nb_heads_mult=1),  # to train: 10Go
    "11.8M": dict(depth=6, head_dim=384, context_size=4096, nb_heads_mult=1),
    "20.5M": dict(depth=6, head_dim=512, context_size=4096, nb_heads_mult=5),
}
