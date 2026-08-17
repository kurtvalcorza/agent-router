from agent_router import AnthropicPricingSource, OpenAIModelPricingSource


ANTHROPIC_HTML = """
<html><body>
<table>
<tr><th>Model</th><th>Base Input Tokens</th><th>5m Cache Writes</th><th>1h Cache Writes</th><th>Cache Hits &amp; Refreshes</th><th>Output Tokens</th></tr>
<tr><td>Claude Sonnet 4</td><td>$3 / MTok</td><td>$3.75 / MTok</td><td>$6 / MTok</td><td>$0.30 / MTok</td><td>$15 / MTok</td></tr>
</table>
<table>
<tr><th>Model</th><th>Batch input</th><th>Batch output</th></tr>
<tr><td>Claude Sonnet 4</td><td>$1.50 / MTok</td><td>$7.50 / MTok</td></tr>
</table>
<table>
<tr><th>≤ 200K input tokens</th><th>&gt; 200K input tokens</th></tr>
<tr><td>Input: $3 / MTok Output: $15 / MTok</td><td>Input: $6 / MTok Output: $22.50 / MTok</td></tr>
</table>
</body></html>
"""

OPENAI_HTML = """
<html><body>
<h1>GPT-5.6 Sol</h1>
<section>Text tokens Per 1M tokens Input $5.00 Cached input $0.50 Output $30.00</section>
<p>Prompts with &gt;272K input tokens are priced at 2x input and 1.5x output for the full request.</p>
<p>Cache writes are billed at 1.25x the uncached input token rate.</p>
</body></html>
"""


def test_anthropic_pricing_source_parses_official_table_shape():
    source = AnthropicPricingSource(
        {"Claude Sonnet 4": "claude-sonnet-4"},
        fetch_text=lambda url: ANTHROPIC_HTML,
    )

    records = source.fetch()

    assert len(records) == 1
    record = records[0]
    assert record.provider == "anthropic"
    assert record.model_id == "claude-sonnet-4"
    assert record.pricing.standard_input == 3.0
    assert record.pricing.standard_output == 15.0
    assert record.pricing.cached_input == 0.30
    assert record.pricing.cache_write == 3.75
    assert record.pricing.batch_input == 1.50
    assert record.pricing.batch_output == 7.50
    assert record.provenance.content_hash


def test_anthropic_pricing_source_applies_reviewed_long_context_rule():
    source = AnthropicPricingSource(
        {"Claude Sonnet 4": "claude-sonnet-4"},
        long_context_thresholds={"Claude Sonnet 4": 200_000},
        fetch_text=lambda url: ANTHROPIC_HTML,
    )

    pricing = source.fetch()[0].pricing

    assert pricing.long_context_threshold == 200_000
    assert pricing.long_context_input == 6.0
    assert pricing.long_context_output == 22.50


def test_openai_pricing_source_parses_model_page_and_long_context_rule():
    source = OpenAIModelPricingSource(
        {"gpt-5.6-sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol"},
        fetch_text=lambda url: OPENAI_HTML,
    )

    record = source.fetch()[0]
    pricing = record.pricing

    assert record.provider == "openai"
    assert record.model_id == "gpt-5.6-sol"
    assert pricing.standard_input == 5.0
    assert pricing.cached_input == 0.5
    assert pricing.standard_output == 30.0
    assert pricing.cache_write == 6.25
    assert pricing.long_context_threshold == 272_000
    assert pricing.long_context_input == 10.0
    assert pricing.long_context_output == 45.0
    assert record.provenance.source.endswith("/gpt-5.6-sol")
