import pytest

from agent_router.pricing import PricingProfile


def test_standard_pricing():
    pricing = PricingProfile(standard_input=1.0, standard_output=4.0)
    assert pricing.estimate(input_tokens=1_000_000, output_tokens=500_000) == 3.0


def test_cache_and_batch_pricing():
    pricing = PricingProfile(
        standard_input=2.0,
        standard_output=8.0,
        cached_input=0.2,
        cache_write=2.5,
        batch_input=1.0,
        batch_output=4.0,
    )
    assert pricing.estimate(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=400_000,
        cache_write_tokens=100_000,
    ) == pytest.approx(1.53)
    assert pricing.estimate(
        input_tokens=1_000_000,
        output_tokens=100_000,
        batch=True,
    ) == pytest.approx(1.4)


def test_long_context_pricing():
    pricing = PricingProfile(
        standard_input=1.0,
        standard_output=3.0,
        long_context_input=2.0,
        long_context_output=6.0,
        long_context_threshold=100_000,
    )
    assert pricing.estimate(input_tokens=100_001, output_tokens=10_000) == pytest.approx(
        0.260002
    )


def test_cache_tokens_cannot_exceed_input():
    pricing = PricingProfile()
    with pytest.raises(ValueError):
        pricing.estimate(input_tokens=10, cached_input_tokens=11)
