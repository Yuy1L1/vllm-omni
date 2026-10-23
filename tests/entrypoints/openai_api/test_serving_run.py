# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import msgspec
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine.messages import NextStageInputMessage, OutputMessage
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.protocol.run import RunRequest
from vllm_omni.entrypoints.openai.serving_run import ServingRun, decode_output, decode_stage_input, encode_payload
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROMPT = {"prompt": "hi"}


def _next_stage_input(requests: list[EngineCoreRequest] | None = None) -> NextStageInputMessage:
    return NextStageInputMessage(
        request_id="req-run",
        source_stage_id=0,
        receiver_stage_id=1,
        requests=requests or [],
        submit_kwargs=None,
        stage_output=None,
        prompt=_PROMPT,
        sampling_params_list=[SamplingParams(), SamplingParams()],
        final_stage_id=1,
        final_output_stage_ids=[1],
    )


def _engine_client(response: NextStageInputMessage | OutputMessage | None = None) -> SimpleNamespace:
    """A two-stage LLM engine client whose run calls return ``response`` (by default, a stage 1 input)."""
    response = response or _next_stage_input()
    return SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="llm")] * 2,
        run_entry_stage=AsyncMock(return_value=response),
        run_downstream_stage=AsyncMock(return_value=response),
    )


def _run_app(engine_client: SimpleNamespace | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.run_serving = ServingRun(engine_client or _engine_client())
    return app


def _run_client(engine_client: SimpleNamespace | None = None) -> TestClient:
    """A client for an app with the API router, serving /v1/run with ``engine_client``."""
    return TestClient(_run_app(engine_client))


def test_stage_input_round_trips_llm_receiver():
    """Ensure an LLM receiver's requests, including omni-only fields, survive encoding."""
    request = build_engine_core_request_from_tokens("req-run", {"prompt_token_ids": [7]}, SamplingParams(max_tokens=9))
    request.additional_information = serialize_additional_information({"codes": torch.arange(2)})
    stage_input = _next_stage_input([request])
    assert decode_stage_input(encode_payload(stage_input), ("llm", "llm")) == stage_input


def test_stage_input_round_trips_sampling_params():
    """Ensure a stage's sampling params (including complex types) survive encoding."""
    stage_input = msgspec.structs.replace(
        _next_stage_input(),
        sampling_params_list=[
            SamplingParams(structured_outputs=StructuredOutputsParams(json={"type": "object"})),
            OmniDiffusionSamplingParams(
                lora_request=LoRARequest(lora_name="adapter", lora_int_id=1, lora_path="/adapter"),
            ),
        ],
    )
    decoded = decode_stage_input(encode_payload(stage_input), ("llm", "diffusion"))
    assert decoded.sampling_params_list == stage_input.sampling_params_list


@pytest.mark.asyncio
async def test_call_runs_stage_with_final_only_output_kind():
    """Ensure a call runs its stage with FINAL_ONLY params so the final output is complete."""
    engine_client = _engine_client()
    await ServingRun(engine_client).run(RunRequest(stage_input=_PROMPT), request_id="run-entry")
    sampling_params_list = engine_client.run_entry_stage.call_args.args[1]
    assert all(params.output_kind == RequestOutputKind.FINAL_ONLY for params in sampling_params_list)


@pytest.mark.asyncio
async def test_call_forwards_requested_replica_to_current_stage():
    engine_client = _engine_client()
    serving = ServingRun(engine_client)

    await serving.run(RunRequest(stage_input=_PROMPT, replica_id=1), request_id="run-entry")
    assert engine_client.run_entry_stage.call_args.kwargs["requested_replica_id"] == 1

    stage_input = _next_stage_input()
    await serving.run(
        RunRequest(stage_id=1, stage_input=encode_payload(stage_input), replica_id=0),
        request_id="run-downstream",
    )
    assert engine_client.run_downstream_stage.call_args.kwargs["requested_replica_id"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("replica_id", [None, 0, 1])
async def test_entry_stage_forwards_none_and_nonnegative_replica_ids(replica_id: int | None):
    engine_client = _engine_client()

    await ServingRun(engine_client).run(
        RunRequest(stage_input=_PROMPT, replica_id=replica_id),
        request_id="run-entry",
    )

    assert engine_client.run_entry_stage.call_args.kwargs["requested_replica_id"] == replica_id


@pytest.mark.parametrize("replica_id", ["abc", "1", 1.5, True, -1])
def test_run_request_rejects_non_strict_or_negative_replica_id(replica_id):
    with pytest.raises(ValidationError):
        RunRequest(stage_input=_PROMPT, replica_id=replica_id)


@pytest.mark.asyncio
async def test_non_final_call_returns_next_stage_id_and_stage_input():
    """Ensure a call that yields returns the next stage's id and its encoded input."""
    next_stage_input = _next_stage_input()
    serving = ServingRun(_engine_client(next_stage_input))
    response = await serving.run(RunRequest(stage_input=_PROMPT), request_id="run-entry")
    assert response.stage_id == next_stage_input.receiver_stage_id
    assert decode_stage_input(response.stage_input, serving.stage_types) == next_stage_input


@pytest.mark.asyncio
async def test_final_call_returns_encoded_output():
    """Ensure the final stage's call returns its raw output, encoded with the audio on its completion output."""
    audio = torch.arange(4.0)
    completion = CompletionOutput(index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None)
    completion.multimodal_output = {"audio": audio}
    final_output = RequestOutput(
        request_id="run-final",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    serving = ServingRun(
        _engine_client(OutputMessage(request_id="run-final", stage_id=1, engine_outputs=final_output, finished=True))
    )
    request = RunRequest(stage_id=1, stage_input=encode_payload(_next_stage_input()))
    response = await serving.run(request, request_id="run-final")
    output = decode_output(response.output)
    assert type(output) is RequestOutput
    assert torch.equal(output.outputs[0].multimodal_output["audio"], audio)


@pytest.mark.asyncio
async def test_entry_stage_call_rejects_encoded_stage_input():
    """Ensure stage 0 rejects an encoded stage input instead of running it as a text prompt."""
    engine_client = _engine_client()
    request = RunRequest(stage_id=0, stage_input=encode_payload(_next_stage_input()))
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_entry_stage.assert_not_awaited()


@pytest.mark.asyncio
async def test_downstream_stage_call_rejects_stage_input_built_for_another_stage():
    """Ensure a call is rejected when its stage_input was built for a different stage than its stage_id."""
    engine_client = _engine_client()
    stage_input = _next_stage_input()
    request = RunRequest(stage_id=stage_input.receiver_stage_id + 1, stage_input=encode_payload(stage_input))
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_downstream_stage.assert_not_awaited()


@pytest.mark.asyncio
async def test_downstream_stage_call_rejects_sampling_params():
    """Ensure a downstream call can't set sampling params, since its stage_input already carries them."""
    engine_client = _engine_client()
    request = RunRequest(stage_id=1, stage_input=encode_payload(_next_stage_input()), sampling_params=[{}])
    with pytest.raises(OmniClientError):
        await ServingRun(engine_client).run(request, request_id="run-bad")
    engine_client.run_downstream_stage.assert_not_awaited()


def test_run_route_returns_400_for_malformed_stage_input():
    """Ensure a stage_input that isn't a valid payload is a client error, not a server error."""
    response = _run_client().post("/v1/run", json={"stage_id": 1, "stage_input": "not a payload"})
    assert response.status_code == 400


def test_run_route_keeps_client_error_status():
    """Ensure a client error raised during a run call keeps its status code."""
    status_code = 422
    engine_client = _engine_client()
    engine_client.run_entry_stage.side_effect = OmniClientError("bad prompt", status_code=status_code)
    response = _run_client(engine_client).post("/v1/run", json={"stage_input": _PROMPT})
    assert response.status_code == status_code


@pytest.mark.asyncio
async def test_invalid_explicit_replica_only_fails_that_http_request():
    engine_client = _engine_client()

    async def route_entry(*_args, requested_replica_id, **_kwargs):
        if requested_replica_id == 99:
            raise OmniClientError("replica_id 99 does not exist")
        return _next_stage_input()

    engine_client.run_entry_stage.side_effect = route_entry
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_run_app(engine_client)),
        base_url="http://test",
    ) as client:
        invalid = await client.post("/v1/run", json={"stage_input": _PROMPT, "replica_id": 99})
        valid = await client.post("/v1/run", json={"stage_input": _PROMPT, "replica_id": 0})

    assert invalid.status_code == 400
    assert "replica_id 99 does not exist" in invalid.json()["detail"]
    assert valid.status_code == 200
    assert engine_client.run_entry_stage.await_count == 2


def test_run_route_omits_unset_response_fields():
    """Ensure a non-final response has no output key, so clients can tell it from the final one."""
    body = _run_client().post("/v1/run", json={"stage_input": _PROMPT}).json()
    assert set(body) == {"stage_id", "stage_input"}
