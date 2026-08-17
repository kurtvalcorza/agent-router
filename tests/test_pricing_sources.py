from agent_router import AnthropicPricingSource


HTML = """
<html><body>
<table>
<tr><th>Model</th><th>Base Input Tokens</th><th>5m Cache Writes</th><th>1h Cache Writes</th><th>Cache Hits &amp; Refreshes</th><th>Output Tokens</th></tr>
<tr><td>Claude Sonnet 4</td><td>$3 / MTok</td><td>$3.75 / MTok</td><td>$6 / MTok</td><td>$0.30 / MTok</td><td>$15 / MTok</td></tr>
</table>
<table>
<tr><th>Model</th><th>Batch input</th><th>Batch output</th></tr>
<tr><td>Claude Sonnet 4</td><td>$1.50 / MTok</td><td>$7.50 / MTok</td></tr>
</table>
</body></html>
"""


def test_anthropic_pricing_source_parses_official_table_shape():
    source = AnthropicPricingSource(
        {"Claude Sonnet 4": "claude-sonnet-4"},
        fetch_text=lambda url: HTML,
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
