from __future__ import annotations

from dataclasses import replace

from .adaptive import AdaptivePolicy, PolicyMode
from .benchmark_runtime import execute_task_strategy, fixed_model_executor
from .catalog import ModelCatalog
from .empirical import EmpiricalSelector, EmpiricalSuccessModel
from .empirical_executor import EmpiricalRoutedModelExecutor
from .evaluation import EvaluationCase, EvaluationRun
from .model_executor import RoutedModelExecutor
from .providers import AnthropicMessagesAdapter, OpenAIResponsesAdapter, ProviderInvoker
from .runtime import RouterRuntime
from .types import ExecutionClass, TelemetryEvent


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
    return _run_runtime_strategy(
        cases,
        strategy="router",
        model_executor=model_executor,
        mode=mode,
    )


def run_empirical_strategy(
    cases: tuple[EvaluationCase, ...],
    *,
    catalog: ModelCatalog,
    success_model: EmpiricalSuccessModel,
    mode: PolicyMode = PolicyMode.BALANCED,
    invoke: ProviderInvoker | None = None,
    recovery_cost_multiplier: float = 1.0,
) -> tuple[EvaluationRun, ...]:
    invoker = invoke or provider_invoker_from_catalog(catalog)
    selector = EmpiricalSelector(
        registry=catalog.registry(),
        success_model=success_model,
        recovery_cost_multiplier=recovery_cost_multiplier,
    )
    model_executor = EmpiricalRoutedModelExecutor(
        selector=selector,
        invoke=invoker,
        adaptive_policy=AdaptivePolicy(mode),
    )
    return _run_runtime_strategy(
        cases,
        strategy="empirical-router",
        model_executor=model_executor,
        mode=mode,
    )


def _run_runtime_strategy(
    cases: tuple[EvaluationCase, ...],
    *,
    strategy: str,
    model_executor,
    mode: PolicyMode,
) -> tuple[EvaluationRun, ...]:
    def execute_task(task):
        events: list[TelemetryEvent] = []
        runtime = RouterRuntime(telemetry=events.append, max_attempts=4)
        initial_class = runtime.policy.route(task).execution_class
        runtime.register_executor(ExecutionClass.LIGHT_REASONING, model_executor)
        runtime.register_executor(ExecutionClass.DEEP_REASONING, model_executor)
        result = runtime.execute(task)
        terminal_class = events[-1].execution_class if events else initial_class
        escalation_rank = {
            ExecutionClass.DETERMINISTIC: 0,
            ExecutionClass.RETRIEVAL: 0,
            ExecutionClass.LIGHT_REASONING: 1,
            ExecutionClass.DEEP_REASONING: 2,
            ExecutionClass.HUMAN_REVIEW: 3,
        }
        escalations = max(
            0,
            escalation_rank[terminal_class] - escalation_rank[initial_class],
        )
        metadata = dict(result.metadata)
        metadata["escalations"] = escalations
        metadata["policy_mode"] = mode.value
        metadata["initial_execution_class"] = initial_class.value
        metadata["terminal_execution_class"] = terminal_class.value
        return replace(result, metadata=metadata)

    return execute_task_strategy(
        cases,
        strategy=strategy,
        execute_task=execute_task,
    )
