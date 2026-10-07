# SPDX-License-Identifier: Apache-2.0
"""MetalKVConnector keeps upstream's step methods working without torch caches."""

from __future__ import annotations

from types import SimpleNamespace

import vllm.v1.worker.gpu.kv_connector as upstream

import vllm_metal.v1.kv_connector as metal_kv_connector
from vllm_metal.v1.kv_connector import MetalKVConnector


def _group(registered: list) -> SimpleNamespace:
    return SimpleNamespace(
        register_kv_caches=registered.append,
        set_host_xfer_buffer_ops=lambda op: None,
    )


def test_sets_the_same_state_as_upstream(monkeypatch):
    """The step methods read state upstream's __init__ sets. If upstream adds
    a field, this fails rather than the step methods at serve time."""
    group = _group([])
    monkeypatch.setattr(upstream, "get_kv_transfer_group", lambda: group)
    monkeypatch.setattr(metal_kv_connector, "get_kv_transfer_group", lambda: group)
    vllm_config = SimpleNamespace()

    reference = upstream.ActiveKVConnector(vllm_config, {})
    metal = MetalKVConnector(vllm_config)

    assert vars(metal) == vars(reference)


def test_does_not_register_torch_caches(monkeypatch):
    """The runner registers KVCacheStorage itself."""
    registered: list = []
    group = _group(registered)
    monkeypatch.setattr(metal_kv_connector, "get_kv_transfer_group", lambda: group)

    MetalKVConnector(SimpleNamespace())

    assert registered == []
