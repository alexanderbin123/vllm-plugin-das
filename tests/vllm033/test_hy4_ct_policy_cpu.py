# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 checkpoint admission; no device/provider availability is implied."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hcu_vllm_das.quantization.hy4_ct_policy import accepts_checkpoint


def checkpoint():
    return {'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
            'config_groups': {'experts': {
                'weights': {'num_bits': 4, 'type': 'int', 'strategy': 'channel',
                            'symmetric': True, 'dynamic': False},
                'input_activations': {'num_bits': 8, 'type': 'int', 'strategy': 'token',
                                      'symmetric': True, 'dynamic': True}}}}


@pytest.mark.parametrize('choice', [None, 'compressed-tensors', 'hcu_hy4_ct_w4a8'])
def test_compatible_hy4_backbone_or_mtp_config_is_selected(choice):
    hf = SimpleNamespace(model_type='hy_v4')
    config = checkpoint()
    before = deepcopy(config)
    assert accepts_checkpoint(config, choice, hf)
    assert config == before


@pytest.mark.parametrize('model_type', ['deepseek_v41', 'hy_v3', 'hy_v4_vl', None])
def test_other_model_families_are_not_claimed(model_type):
    assert not accepts_checkpoint(checkpoint(), None, SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize('side,key,value', [
    ('weights', 'num_bits', 8), ('weights', 'type', 'float'),
    ('weights', 'strategy', 'group'), ('weights', 'symmetric', False),
    ('weights', 'dynamic', True), ('weights', 'group_size', 128),
    ('weights', 'block_structure', [128, 128]),
    ('input_activations', 'num_bits', 4), ('input_activations', 'dynamic', False),
    ('input_activations', 'type', 'float'), ('input_activations', 'symmetric', False),
])
def test_unsupported_quantization_is_not_selected(side, key, value):
    config = checkpoint()
    config['config_groups']['experts'][side][key] = value
    assert not accepts_checkpoint(config, None, SimpleNamespace(model_type='hy_v4'))


@pytest.mark.parametrize('config', [None, {}, {'config_groups': []}, {'quant_method': 'fp8'}])
def test_malformed_or_different_checkpoint_is_not_claimed(config):
    assert not accepts_checkpoint(config, None, SimpleNamespace(model_type='hy_v4'))


def test_explicit_other_quantizer_and_nonpacked_weights_are_preserved():
    hf = SimpleNamespace(model_type='hy_v4')
    config = checkpoint()
    assert not accepts_checkpoint(config, 'fp8', hf)
    config['format'] = 'dense'
    assert not accepts_checkpoint(config, None, hf)


def test_real_hy4_config_delegates_without_claiming_other_models(monkeypatch):
    """Only unavailable framework/base imports are doubled, not HY4 policy."""
    import importlib
    import sys
    from types import ModuleType
    calls = []
    current = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='hy_v4')))
    fake_config = ModuleType('vllm.config')
    fake_config.get_current_vllm_config = lambda: current
    fake_base = ModuleType('hcu_vllm_das.quantization.ct_w4a8')
    class Base:
        def get_quant_method(self, layer, prefix):
            calls.append((layer, prefix))
            return 'delegated'
    fake_base.HcuDSV41CTW4A8Config = Base
    platform = ModuleType('hcu_vllm_das.quantization.dsv41_channel_fp8')
    platform._is_hcu = lambda: True
    health = ModuleType('hcu_vllm_das.hy4_quant_activation')
    health.check_health = lambda: calls.append('health')
    for name, value in {'vllm.config': fake_config,
        fake_base.__name__: fake_base, platform.__name__: platform,
        health.__name__: health}.items():
        monkeypatch.setitem(sys.modules, name, value)
    name = 'hcu_vllm_das.quantization.hy4_ct_w4a8'
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    cls = module.HcuHY4CTW4A8Config
    try:
        assert cls.override_quantization_method(checkpoint(), None, current.model_config.hf_config) == cls.get_name()
        assert calls == ['health']
        assert cls().get_quant_method('layer', 'model.layers.1') == 'delegated'
        assert calls[-1] == ('layer', 'model.layers.1')
        current.model_config.hf_config.model_type = 'deepseek_v41'
        assert cls.override_quantization_method(checkpoint(), None, current.model_config.hf_config) is None
        with pytest.raises(ValueError, match='HY4'):
            cls().get_quant_method('layer', 'other')
    finally:
        sys.modules.pop(name, None)


def test_hy4_activation_follows_the_adapter_ct_registration(monkeypatch):
    """Use the real loader and descriptor priorities; publication is a double."""
    from dataclasses import replace
    import importlib
    from hcu_adapter.extensions import ExtensionContext, ExtensionStatus, activate_extensions
    from hcu_vllm_das.adapter_api import adapter_module

    provider = importlib.reload(importlib.import_module('hcu_vllm_das.hy4_quant_activation'))
    ct = adapter_module('ct_w4a8_registration')
    events = []
    primary = replace(ct.extension_spec(),
                      activate=lambda context: events.append('ct-published'),
                      check_health=None)
    monkeypatch.setattr(provider, 'adapter_module', lambda name: SimpleNamespace(
        set_additional_config_provider=lambda name, factory: None))

    def require_publication():
        assert events == ['ct-published']
        events.append('hy4-checked')

    monkeypatch.setattr(provider, 'check_health', require_publication)
    points = [
        SimpleNamespace(name='das_hcu_hy4_ct_w4a8', load=lambda: provider.extension_spec),
        SimpleNamespace(name='das_hcu_ct_w4a8', load=lambda: lambda: primary),
    ]
    context = ExtensionContext('0.5.22', '0.33.0',
                               requested_capabilities={'hy4-ct-w4a8'})
    reports = activate_extensions(context, entry_points=points)
    assert len(reports) == 2
    assert all(report.status is ExtensionStatus.ACTIVE for report in reports)
    assert events == ['ct-published', 'hy4-checked']
