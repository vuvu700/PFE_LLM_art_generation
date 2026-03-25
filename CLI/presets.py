"""
pour ajouter un preset il suffit de rajouter une ligne avec les differents parametres
"""

PRESETS = {
    "1.6M": dict(depth=6, head_dim=128, context_size=4096, nb_heads_mult=1),  # 1.8Go
    "1.6M_ext": dict(
        depth=6, head_dim=128, context_size=4096 * 4, nb_heads_mult=1
    ),  # 7.6Go
    "5.5M": dict(depth=6, head_dim=256, context_size=4096, nb_heads_mult=1),  # 2.8Go
    "5.5M_ext": dict(
        depth=6, head_dim=256, context_size=4096 * 4, nb_heads_mult=1
    ),  # 11.5Go
    "6.0M": dict(depth=7, head_dim=128, context_size=4096, nb_heads_mult=2),  # 2.8Go
    "11.8M": dict(depth=6, head_dim=384, context_size=4096, nb_heads_mult=1),  # 4.0Go
    "20.5M": dict(depth=6, head_dim=512, context_size=4096, nb_heads_mult=1),  # ...
}
