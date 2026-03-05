from __future__ import annotations

from typing import Any, Literal, TypedDict

CommandName = Literal["servebench", "enginebench", "compare-runs"]


class TaskSpec(TypedDict):
    command: CommandName
    base_args: dict[str, Any]


class ResourceSpec(TypedDict):
    enabled: bool
    interval_sec: float


class MlflowSpec(TypedDict):
    enabled: bool
    required: bool
    uri: str | None
    experiment_name: str


class TrackingSpec(TypedDict):
    mlflow: MlflowSpec


class RgpSpec(TypedDict):
    enabled: bool
    root_glob: str


class ProfilingSpec(TypedDict):
    rgp: RgpSpec


class GateSpec(TypedDict):
    min_success_ratio: float
    require_artifacts_ready: bool


class ExpSpec(TypedDict):
    version: Literal["exp_spec_v1"]
    name: str
    task: TaskSpec
    matrix: dict[str, list[Any]]
    resources: ResourceSpec
    tracking: TrackingSpec
    profiling: ProfilingSpec
    gates: GateSpec

