PATH={{ bin_dir }}"${PATH:+:$PATH}"
export PATH
{% if let Some(uv) = uv %}UV_CACHE_DIR={{ uv.cache }}
UV_PYTHON_INSTALL_DIR={{ uv.python_installs }}
UV_PYTHON_PREFERENCE={{ uv.python_preference }}
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_PREFERENCE
{% endif %}{{ script }}
