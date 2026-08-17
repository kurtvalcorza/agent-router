from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import monotonic

from .policy import RoutingPolicy, next_execution_class
from .types import (
    Budget,
    BudgetExceeded,
    ExecutionClass,
    ExecutionContext,
    ExecutionResult,
    RouteDecision,
    Task,
    TelemetryEvent,
    Verification,
    VerificationStatus,
)

Executor = Callable[[Task, ExecutionContext], ExecutionResult]
Verifier = Callable[[Task, ExecutionResult], Verification]
TelemetrySink = Callable[[TelemetryEvent], None]


def _accept(_: Task, __: ExecutionResult) -> Verification:
    return Verification(VerificationStatus.PASS)


def _noop(_: TelemetryEvent) -> None:
    return None


class RouterRuntime:
    def __init__(
        self,
        *,
        policy: RoutingPolicy | None = None,
        verifier: Verifier | None = None,
        telemetry: TelemetrySink | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.policy = policy or RoutingPolicy()
        self.verifier = verifier or _accept
        self.telemetry = telemetry or _noop
        self.max_attempts = max_attempts
        self._executors: dict[ExecutionClass, Executor] = {}

    def register_executor(self, execution_class: ExecutionClass, executor: Executor) -> None:
        self._executors[execution_class] = executor

    def execute(self, task: Task, *, budget: Budget | None = None) -> ExecutionResult:
        budget = budget or Budget()
        started = monotonic()
        decision = self.policy.route(task)
        attempt = 1

        while attempt <= self.max_attempts:
            self._check_latency(budget, started)
            executor = self._executors.get(decision.execution_class)
            if executor is None:
                decision = self._escalate(decision, "no executor registered")
                attempt += 1
                continue

            context = ExecutionContext(budget=budget, attempt=attempt, decision=decision)
            result = executor(task, context)
            budget.consume(
                cost_usd=result.cost_usd,
                model_calls=result.model_calls,
                tool_calls=result.tool_calls,
            )

            verification = self.verifier(task, result)
            self.telemetry(
                TelemetryEvent(
                    task_kind=task.kind,
                    execution_class=decision.execution_class,
                    route_reason=decision.reason,
                    attempt=attempt,
                    verification=verification.status,
                    cost_usd=result.cost_usd,
                    model_calls=result.model_calls,
                    tool_calls=result.tool_calls,
                )
            )

            if verification.status is VerificationStatus.PASS:
                return result

            if verification.status is VerificationStatus.RETRY:
                attempt += 1
                continue

            decision = self._escalate(
                decision,
                verification.reason or "verification requested escalation",
            )
            attempt += 1

        raise RuntimeError("execution exhausted retry/escalation budget")

    @staticmethod
    def _escalate(decision: RouteDecision, reason: str) -> RouteDecision:
        return replace(
            decision,
            execution_class=next_execution_class(decision.execution_class),
            reason=f"{decision.reason}; escalated: {reason}",
        )

    @staticmethod
    def _check_latency(budget: Budget, started: float) -> None:
        elapsed = monotonic() - started
        if budget.max_latency_seconds is not None and elapsed > budget.max_latency_seconds:
            raise BudgetExceeded("latency budget exceeded")
