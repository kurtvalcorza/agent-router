from types import SimpleNamespace

from agent_router import AnthropicInventoryFetcher, OpenAIInventoryFetcher, SourceProvenance


class FakeOpenAIModels:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-test-b", object="model", created=2, owned_by="openai"),
                SimpleNamespace(id="gpt-test-a", object="model", created=1, owned_by="openai"),
            ]
        )


class FakeOpenAIClient:
    models = FakeOpenAIModels()


class FakeAnthropicModels:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="claude-test-b",
                    display_name="Claude Test B",
                    created_at="2026-01-02T00:00:00Z",
                    type="model",
                ),
                SimpleNamespace(
                    id="claude-test-a",
                    display_name="Claude Test A",
                    created_at="2026-01-01T00:00:00Z",
                    type="model",
                ),
            ]
        )


class FakeAnthropicClient:
    models = FakeAnthropicModels()


def test_openai_inventory_fetcher_normalizes_and_sorts_models():
    records = OpenAIInventoryFetcher(client=FakeOpenAIClient()).fetch()

    assert [record.model_id for record in records] == ["gpt-test-a", "gpt-test-b"]
    assert all(record.provider == "openai" for record in records)
    assert all(record.available for record in records)
    assert records[0].created_at == 1
    assert records[0].owned_by == "openai"
    assert records[0].provenance is not None
    assert records[0].provenance.source == "https://api.openai.com/v1/models"
    assert len(records[0].provenance.content_hash) == 64


def test_anthropic_inventory_fetcher_normalizes_and_sorts_models():
    records = AnthropicInventoryFetcher(client=FakeAnthropicClient()).fetch()

    assert [record.model_id for record in records] == ["claude-test-a", "claude-test-b"]
    assert all(record.provider == "anthropic" for record in records)
    assert all(record.owned_by == "anthropic" for record in records)
    assert records[0].metadata["display_name"] == "Claude Test A"
    assert records[0].metadata["created_at"] == "2026-01-01T00:00:00Z"
    assert records[0].provenance.source == "https://api.anthropic.com/v1/models"


def test_provenance_hash_is_stable_for_same_payload():
    left = SourceProvenance.from_payload(
        source="test",
        payload={"b": 2, "a": 1},
        retrieved_at="2026-08-17T00:00:00+00:00",
    )
    right = SourceProvenance.from_payload(
        source="test",
        payload={"a": 1, "b": 2},
        retrieved_at="2026-08-17T00:00:00+00:00",
    )

    assert left.content_hash == right.content_hash
