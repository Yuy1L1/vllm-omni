# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.distributed.omni_coordinator import ReplicaInfo, ReplicaStatus
from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool, StageUnavailableError
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStageClient:
    final_output = False

    def __init__(
        self,
        input_addr: str,
        *,
        stage_type: str = "llm",
        submit_error: Exception | None = None,
    ) -> None:
        self.stage_type = stage_type
        self.client_addresses = {"input_address": input_addr}
        self.request_address = input_addr if stage_type == "diffusion" else None
        self.submit_error = submit_error
        self.add_request_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.abort_calls: list[list[str]] = []

    async def add_request_async(self, *args: Any, **kwargs: Any) -> None:
        self.add_request_calls.append((args, kwargs))
        if self.submit_error is not None:
            raise self.submit_error

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(request_ids)


class FakeOutputProcessor:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []

    def add_request(self, request: Any, **_kwargs: Any) -> None:
        self.added.append(request.request_id)

    def remove_request(self, request_id: str) -> None:
        self.removed.append(request_id)

    def abort_requests(self, request_ids: list[str], *, internal: bool) -> list[str]:
        assert internal is False
        return request_ids


class FakeHub:
    def __init__(self, replicas: list[ReplicaInfo]) -> None:
        self.replicas = replicas

    def get_replicas_for_stage(self, stage_id: int) -> SimpleNamespace:
        return SimpleNamespace(replicas=[replica for replica in self.replicas if replica.stage_id == stage_id])


class RoundRobinLB:
    def __init__(self) -> None:
        self.next = 0

    def select(self, _task: Any, replicas: list[ReplicaInfo]) -> int:
        selected = self.next % len(replicas)
        self.next += 1
        return selected


def replica(input_addr: str, *, status: ReplicaStatus = ReplicaStatus.UP) -> ReplicaInfo:
    return ReplicaInfo(
        input_addr=input_addr,
        output_addr=f"{input_addr}-output",
        stage_id=0,
        status=status,
        queue_length=0,
        last_heartbeat=0.0,
        registered_at=0.0,
    )


def distributed_pool(
    statuses: tuple[ReplicaStatus, ...] = (ReplicaStatus.UP, ReplicaStatus.UP),
) -> tuple[StagePool, list[FakeStageClient], FakeHub]:
    clients = [FakeStageClient(f"tcp://replica-{index}") for index in range(len(statuses))]
    hub = FakeHub(
        [replica(f"tcp://replica-{index}", status=status) for index, status in enumerate(statuses)]
    )
    pool = StagePool(0, clients, output_processor=FakeOutputProcessor())
    pool.attach_hub(hub)
    pool.attach_load_balancer(RoundRobinLB())
    return pool, clients, hub


def request_state(request_id: str) -> OrchestratorRequestState:
    return OrchestratorRequestState(
        request_id=request_id,
        sampling_params_list=[SamplingParams(max_tokens=1)],
        final_stage_id=0,
    )


def request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, prompt_token_ids=[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_id", [0, 1])
async def test_explicit_id_selects_exact_replica(requested_id: int) -> None:
    pool, _, _ = distributed_pool()

    selected = await pool._pick_or_select("req", requested_replica_id=requested_id)

    assert selected == requested_id
    assert pool.get_bound_replica_id("req") == requested_id


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_id", [-1, 99])
async def test_invalid_explicit_id_is_rejected_without_binding(requested_id: int) -> None:
    pool, clients, _ = distributed_pool()

    with pytest.raises(OmniClientError, match=rf"replica_id {requested_id} does not exist"):
        await pool._pick_or_select("req", requested_replica_id=requested_id)

    assert pool.get_bound_replica_id("req") is None
    assert all(not client.add_request_calls for client in clients)


@pytest.mark.asyncio
async def test_explicit_down_replica_is_rejected_without_fallback() -> None:
    pool, _, _ = distributed_pool((ReplicaStatus.UP, ReplicaStatus.DOWN))

    with pytest.raises(OmniClientError, match="replica_id 1 is not UP and attached"):
        await pool._pick_or_select("req", requested_replica_id=1)

    assert pool.get_bound_replica_id("req") is None


@pytest.mark.asyncio
async def test_none_preserves_automatic_round_robin() -> None:
    pool, _, _ = distributed_pool()

    assert await pool._pick_or_select("req-0", requested_replica_id=None) == 0
    assert await pool._pick_or_select("req-1", requested_replica_id=None) == 1


@pytest.mark.asyncio
async def test_automatic_routing_skips_down_replica() -> None:
    pool, _, _ = distributed_pool((ReplicaStatus.DOWN, ReplicaStatus.UP))

    assert await pool._pick_or_select("req") == 1


@pytest.mark.asyncio
async def test_single_available_replica_works_for_automatic_and_explicit() -> None:
    pool, _, _ = distributed_pool((ReplicaStatus.DOWN, ReplicaStatus.UP))

    assert await pool._pick_or_select("auto") == 1
    assert await pool._pick_or_select("explicit", requested_replica_id=1) == 1


@pytest.mark.asyncio
async def test_no_serviceable_replica_fails_without_unbounded_wait() -> None:
    pool, _, _ = distributed_pool((ReplicaStatus.DOWN, ReplicaStatus.DOWN))
    pool.DISPATCH_WAIT_TIMEOUT_S = 0

    with pytest.raises(StageUnavailableError, match="no UP replica"):
        await pool._pick_or_select("req")


@pytest.mark.asyncio
async def test_explicit_preselection_and_submit_use_same_replica() -> None:
    pool, clients, _ = distributed_pool()
    req = request("req")

    assert await pool._pick_or_select("req", requested_replica_id=1) == 1
    assert pool.preselect_replica_id("req") == 1
    assert await pool.submit_initial("req", request_state("req"), req) == 1

    assert not clients[0].add_request_calls
    assert len(clients[1].add_request_calls) == 1


@pytest.mark.asyncio
async def test_automatic_preselection_and_submit_remain_sticky() -> None:
    pool, clients, _ = distributed_pool()

    assert pool.preselect_replica_id("req") == 0
    assert await pool.submit_initial("req", request_state("req"), request("req")) == 0

    assert len(clients[0].add_request_calls) == 1
    assert not clients[1].add_request_calls


@pytest.mark.asyncio
async def test_repeated_explicit_selection_does_not_switch_replica() -> None:
    pool, _, _ = distributed_pool()

    assert await pool._pick_or_select("req", requested_replica_id=1) == 1
    assert await pool._pick_or_select("req") == 1
    assert await pool._pick_or_select("req", requested_replica_id=1) == 1


@pytest.mark.asyncio
async def test_existing_binding_conflict_is_rejected() -> None:
    pool, _, _ = distributed_pool()
    await pool._pick_or_select("req", requested_replica_id=0)

    with pytest.raises(OmniClientError, match="already bound.*replica-0.*replica-1"):
        await pool._pick_or_select("req", requested_replica_id=1)

    assert pool.get_bound_replica_id("req") == 0


@pytest.mark.asyncio
async def test_explicit_child_selection_conflicting_with_parent_affinity_is_rejected() -> None:
    pool, _, _ = distributed_pool()
    await pool._pick_or_select("parent", requested_replica_id=0)

    with pytest.raises(OmniClientError, match="inherits.*parent.*replica-0.*replica-1"):
        await pool._pick_or_select(
            "child",
            affinity_request_id="parent",
            requested_replica_id=1,
        )

    assert pool.get_bound_replica_id("child") is None


@pytest.mark.asyncio
async def test_failed_explicit_selection_leaves_no_binding() -> None:
    pool, _, _ = distributed_pool((ReplicaStatus.UP, ReplicaStatus.DOWN))

    with pytest.raises(OmniClientError):
        await pool._pick_or_select("req", requested_replica_id=1)

    assert pool.get_bound_replica_id("req") is None
    assert "req" not in pool._requested_replica_ids


@pytest.mark.asyncio
async def test_explicit_binding_is_released_after_request() -> None:
    pool, _, _ = distributed_pool()
    await pool._pick_or_select("req", requested_replica_id=0)

    pool.release_binding("req")

    assert pool.get_bound_replica_id("req") is None
    assert "req" not in pool._requested_replica_ids


@pytest.mark.asyncio
async def test_llm_dispatch_exception_releases_binding_and_output_state() -> None:
    pool, clients, _ = distributed_pool()
    clients[1].submit_error = RuntimeError("submit failed")
    output_processor = pool.output_processor
    await pool._pick_or_select("req", requested_replica_id=1)

    with pytest.raises(RuntimeError, match="submit failed"):
        await pool.submit_initial("req", request_state("req"), request("req"))

    assert output_processor.added == ["req"]
    assert output_processor.removed == ["req"]
    assert pool.get_bound_replica_id("req") is None
    assert "req" not in pool._requested_replica_ids


@pytest.mark.asyncio
async def test_diffusion_dispatch_exception_releases_explicit_binding() -> None:
    addr = "tcp://replica-0"
    client = FakeStageClient(addr, stage_type="diffusion", submit_error=RuntimeError("submit failed"))
    hub = FakeHub([replica(addr)])
    pool = StagePool(0, [client])
    pool.attach_hub(hub)
    pool.attach_load_balancer(RoundRobinLB())
    await pool._pick_or_select("req", requested_replica_id=0)

    with pytest.raises(RuntimeError, match="submit failed"):
        await pool.submit_initial("req", request_state("req"), {"prompt": "hi"})

    assert pool.get_bound_replica_id("req") is None
    assert "req" not in pool._requested_replica_ids


@pytest.mark.asyncio
async def test_abort_uses_bound_explicit_replica_then_binding_can_be_released() -> None:
    pool, clients, _ = distributed_pool()
    await pool._pick_or_select("req", requested_replica_id=1)

    await pool.abort_requests(["req"])
    pool.release_binding("req")

    assert not clients[0].abort_calls
    assert clients[1].abort_calls == [["req"]]
    assert pool.get_bound_replica_id("req") is None


@pytest.mark.asyncio
async def test_hub_up_but_client_not_attached_is_rejected() -> None:
    addr = "tcp://replica-0"
    client = FakeStageClient(addr)
    pool = StagePool(0, [client])
    pool.remove_client(addr)
    pool.attach_hub(FakeHub([replica(addr)]))
    pool.attach_load_balancer(RoundRobinLB())

    with pytest.raises(OmniClientError, match="not UP and attached"):
        await pool._pick_or_select("req", requested_replica_id=0)


@pytest.mark.asyncio
async def test_hub_and_pool_address_mismatch_is_rejected() -> None:
    pool = StagePool(0, [FakeStageClient("tcp://attached")])
    pool.attach_hub(FakeHub([replica("tcp://different")]))
    pool.attach_load_balancer(RoundRobinLB())

    with pytest.raises(OmniClientError, match="not UP and attached"):
        await pool._pick_or_select("req", requested_replica_id=0)


@pytest.mark.asyncio
async def test_sparse_replica_id_is_a_real_id_not_candidate_index() -> None:
    client0 = FakeStageClient("tcp://replica-0")
    client2 = FakeStageClient("tcp://replica-2")
    pool = StagePool(0, [client0])
    pool.add_client("tcp://replica-2", client2, replica_id=2)
    pool.attach_hub(FakeHub([replica("tcp://replica-0"), replica("tcp://replica-2")]))
    pool.attach_load_balancer(RoundRobinLB())

    assert pool.clients[1] is None
    assert await pool._pick_or_select("explicit", requested_replica_id=2) == 2
    assert await pool._pick_or_select("auto-0") == 0
    assert await pool._pick_or_select("auto-1") == 2


@pytest.mark.asyncio
async def test_removing_one_replica_does_not_stop_healthy_replica() -> None:
    pool, _, hub = distributed_pool()
    pool.remove_client("tcp://replica-1")
    hub.replicas = [replica("tcp://replica-0")]

    assert await pool._pick_or_select("req") == 0


@pytest.mark.asyncio
async def test_replica_reregistration_restores_same_stable_id() -> None:
    pool, _, hub = distributed_pool()
    pool.remove_client("tcp://replica-1")
    replacement = FakeStageClient("tcp://replica-1")
    assert pool.add_client("tcp://replica-1", replacement) == 1
    hub.replicas = [replica("tcp://replica-0"), replica("tcp://replica-1")]

    assert await pool._pick_or_select("req", requested_replica_id=1) == 1
    assert pool.get_bound_client("req") is replacement


@pytest.mark.asyncio
async def test_validation_to_dispatch_down_race_does_not_fallback() -> None:
    pool, _, hub = distributed_pool()
    assert await pool._pick_or_select("req", requested_replica_id=1) == 1
    hub.replicas = [replica("tcp://replica-0"), replica("tcp://replica-1", status=ReplicaStatus.DOWN)]

    with pytest.raises(OmniClientError, match="replica_id 1 is not UP and attached"):
        await pool._pick_or_select("req")

    assert pool.get_bound_replica_id("req") == 1


@pytest.mark.asyncio
async def test_dispatch_message_preserves_explicit_intent_after_binding_cleanup() -> None:
    pool, clients, hub = distributed_pool()
    assert await pool._pick_or_select("req", requested_replica_id=1) == 1

    # Model membership cleanup winning the race against the already queued
    # submission. The queued requested_replica_id must still be authoritative.
    pool.release_binding("req")
    hub.replicas = [replica("tcp://replica-0"), replica("tcp://replica-1", status=ReplicaStatus.DOWN)]

    with pytest.raises(OmniClientError, match="replica_id 1 is not UP and attached"):
        await pool.submit_initial(
            "req",
            request_state("req"),
            request("req"),
            requested_replica_id=1,
        )

    assert all(not client.add_request_calls for client in clients)
    assert pool.get_bound_replica_id("req") is None


@pytest.mark.asyncio
async def test_concurrent_explicit_and_automatic_requests_do_not_cross_bindings() -> None:
    pool, _, _ = distributed_pool()

    selected = await asyncio.gather(
        pool._pick_or_select("explicit-0", requested_replica_id=0),
        pool._pick_or_select("explicit-1", requested_replica_id=1),
        pool._pick_or_select("automatic"),
    )

    assert selected[:2] == [0, 1]
    assert pool.get_bound_replica_id("explicit-0") == 0
    assert pool.get_bound_replica_id("explicit-1") == 1
    assert pool.get_bound_replica_id("automatic") == selected[2]


@pytest.mark.asyncio
async def test_request_id_can_be_reused_after_release_without_old_pin() -> None:
    pool, _, _ = distributed_pool()
    await pool._pick_or_select("req", requested_replica_id=0)
    pool.release_binding("req")

    assert await pool._pick_or_select("req", requested_replica_id=1) == 1


@pytest.mark.asyncio
async def test_legacy_mode_supports_explicit_and_automatic_selection() -> None:
    clients = [FakeStageClient("local-0"), FakeStageClient("local-1")]
    pool = StagePool(0, clients)

    assert await pool._pick_or_select("explicit", requested_replica_id=1) == 1
    assert await pool._pick_or_select("automatic") == 0
