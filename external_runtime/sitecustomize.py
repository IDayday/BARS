from __future__ import annotations

try:
    import jax
    from jax._src import config as _jax_internal_config

    if not hasattr(jax.config, "define_bool_state") and hasattr(_jax_internal_config, "define_bool_state"):
        jax.config.define_bool_state = _jax_internal_config.define_bool_state
except Exception:
    pass
