# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real Core replica mapping capture/replay; model and EP owners are doubles."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import static_eplb_runtime as runtime
from replay_v41_static_eplb import (
    EP_SIZE, LOGICAL, PHYSICAL, ROWS, ModelDouble, cfg, load, write_map,
)

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize('model_type', ['hy_v4', 'hy_v4_mtp'])
@pytest.mark.parametrize('policy', ['nearest', 'locality_fair'])
@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('batch', [1, 7, 2056])
def test_core_replica_mapping_graph_replays_changed_inputs(
    tmp_path, model_type, policy, rank, batch,
):
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip('real HCU required')
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        eplb_map_to_physical_and_record,
    )

    path = write_map(tmp_path)
    model = ModelDouble(load(path))
    weights = torch.zeros(1, device='cuda')
    for layer in model.layers:
        layer.routed_experts.get_expert_weights = lambda: [weights]
    config = cfg(expert_map_path=path, static_dispatch_policy=policy)
    config.model_config.hf_config.model_type = model_type
    group = NS(device_group=NS(rank=lambda: rank, size=lambda: EP_SIZE))
    with (patch('vllm.distributed.get_ep_group', return_value=group),
          patch('vllm.distributed.get_node_count', return_value=1),
          patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                return_value=model)):
        runtime.install_routing(model, config)

    states = [layer.eplb_state for layer in model.layers]
    maps = [state.logical_to_physical_map.cpu().tolist() for state in states]
    for state, row, mapping in zip(states, ROWS, maps):
        counts = [row.count(logical) for logical in range(LOGICAL)]
        assert state.logical_replica_count.cpu().tolist() == counts
        assert not bool(state.should_record_tensor.item())
        for logical, count in enumerate(counts):
            assert sorted(mapping[logical][:count]) == [i for i, value in enumerate(row) if value == logical]

    def inputs(step):
        logical = (torch.arange(batch * 8).reshape(batch, 8) + step) % LOGICAL
        logical[-1, -2:] = torch.tensor([-1, LOGICAL])
        return logical

    ids = inputs(0).to(device='cuda')
    input_address = ids.data_ptr()

    def execute():
        return [eplb_map_to_physical_and_record(
            ids, state.expert_load_view, state.logical_to_physical_map,
            state.logical_replica_count, state.should_record_tensor,
        ) for state in states]

    # Compile and allocate outside capture; the captured body is the actual
    # public Core routing function, including input conversion and output writes.
    for _ in range(3):
        execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = execute()
    output_addresses = [out.data_ptr() for out in outputs]

    previous = None
    for step in (1, 2, 3):
        logical = inputs(step)
        ids.copy_(logical)
        graph.replay()
        torch.cuda.synchronize()
        current = []
        for state, row, mapping, out in zip(states, ROWS, maps, outputs):
            # Independent scalar oracle, using the known placement's counts.
            expected = []
            for token, experts in enumerate(logical.tolist()):
                hashed = (token * 2654435769) & 0xFFFFFFFF
                expected.append([
                    mapping[expert][hashed % row.count(expert)]
                    if 0 <= expert < LOGICAL else -1
                    for expert in experts
                ])
            actual = out.cpu()
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=0)
            assert state.expert_load_view.shape == (PHYSICAL,)
            assert torch.count_nonzero(state.expert_load_view).item() == 0
            current.append(actual)
        if previous is not None:
            assert any(not torch.equal(before, after) for before, after in zip(previous, current))
        previous = current
        assert ids.data_ptr() == input_address
        assert [out.data_ptr() for out in outputs] == output_addresses
