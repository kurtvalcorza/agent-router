from agent_router import (
    ExecutionClass,
    Requirement,
    Risk,
    RoutingPolicy,
    Task,
)


def _task(requirements: set[Requirement], risk: Risk = Risk.LOW) -> Task:
    return Task(kind="fetch", payload={}, requirements=requirements, risk=risk)


def test_semantic_reasoning_routes_to_light_reasoning() -> None:
    decision = RoutingPolicy().route(_task({Requirement.SEMANTIC_REASONING}))
    assert decision.execution_class is ExecutionClass.LIGHT_REASONING


def test_plain_external_data_routes_to_retrieval() -> None:
    decision = RoutingPolicy().route(_task({Requirement.EXTERNAL_DATA}))
    assert decision.execution_class is ExecutionClass.RETRIEVAL


def test_high_risk_retrieval_escalates_to_deep_reasoning() -> None:
    # Regression: the retrieval gate used to fire before the high-risk gate, so a high-risk
    # retrieval task was routed to the weakest tier instead of DEEP_REASONING.
    decision = RoutingPolicy().route(_task({Requirement.EXTERNAL_DATA}, risk=Risk.HIGH))
    assert decision.execution_class is ExecutionClass.DEEP_REASONING


def test_high_reliability_retrieval_escalates_to_deep_reasoning() -> None:
    decision = RoutingPolicy().route(
        _task({Requirement.EXTERNAL_DATA, Requirement.HIGH_RELIABILITY})
    )
    assert decision.execution_class is ExecutionClass.DEEP_REASONING


def test_long_context_retrieval_escalates_to_deep_reasoning() -> None:
    decision = RoutingPolicy().route(
        _task({Requirement.EXTERNAL_DATA, Requirement.LONG_CONTEXT})
    )
    assert decision.execution_class is ExecutionClass.DEEP_REASONING
