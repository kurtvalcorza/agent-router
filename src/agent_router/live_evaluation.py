from __future__ import annotations

from dataclasses import replace

from .adaptive import AdaptivePolicy, PolicyMode
from .benchmark_runtime import execute_task_strategy, fixed_model_executor
from .catalog import ModelCatalog
from .evaluation import EvaluationCase, EvaluationRun
from .model_executor import RoutedModelExecutor
from .providers import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
    ProviderInvoker,
)
from .runtime import RouterRuntime
from .types import ExecutionClass, ExecutionResult, TelemetryEvent


def provider_invoker_from_catalog(catalog: ModelCatalog) -> ProviderInvoker:
    providers = {profile.provider for profile in catalog.profiles}
    adapters = {}
    if "openai" in providers:
        adapters["openai"] = OpenAIResponsesAdapter.from_env()
    if "anthropic" in providers:
        adapters["anthropic"] = AnthropicMessagesAdapter.from_env()
    unsupported = providers - {"openai", "anthropic"}
    if unsupported:
        raise RuntimeError(
            "live evaluation has no provider adapters for: " + ", ".join(sorted(unsupported))
        )
    return ProviderInvoker(adapters)


def run_fixed_baseline(
    cases: tuple[EvaluationCase, ...],
    *,
    strategy: str,
    catalog: ModelCatalog,
    model: str,
    invoke: ProviderInvoker | None = None,
) -> tuple[EvaluationRun, ...]:
    invoker = invoke or provider_invoker_from_catalog(catalog)
    profile = catalog.registry().get(model)
    return execute_task_strategy(
        cases,
        strategy=strategy,
        execute_task=fixed_model_executor(profile, invoker),
    )


def run_router_strategy(
    cases: tuple[EvaluationCase, ...],
    *,
    catalog: ModelCatalog,
    mode: PolicyMode = PolicyMode.BALANCED,
    invoke: ProviderInvoker | None = None,
) -> tuple[EvaluationRun, ...]:
    invoker = invoke or provider_invoker_from_catalog(catalog)
    registry = catalog.registry()
    model_executor = RoutedModelExecutor(
        registry=registry,
        invoke=invoker,
        adaptive_policy=AdaptivePolicy(mode),
    )

    def execute_task(task):
        events: list[TelemetryEvent] = []
        runtime = RouterRuntime(telemetry=events.append, max_attempts=4)
        runtime.register_executor(ExecutionClass.LIGHT_REASONING, model_executor)
        runtime.register_executor(ExecutionClass.DEEP_REASONING, model_executor)
        result = runtime.execute(task)
        escalations = max(0, len(events) - 1)
        if events:
            first = events[0].execution_class
            last = events[-1].execution_class
            escalations = int(first != last) + max(0, len(events) - 1)
        metadata = dict(result.metadata)
        metadata["escalations"] = escalations
        metadata["policy_mode"] = mode.value
        return replace(result, metadata=metadata)

    return execute_task_strategy(
        cases,
        strategy="router",
        execute_task=execute_task,
    )
