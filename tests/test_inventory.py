from types import SimpleNamespace

from agent_router import OpenAIInventoryFetcher, SourceProvenance


class FakeModels:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="gpt-test-b",
                    object="model",
                    created=2,
                    owned_by="openai",
                ),
                SimpleNamespace(
                    id="gpt-test-a",
                    object="model",
                    created=1,
                    owned_by="openai",
                ),
            ]
        )


class FakeClient:
    models = FakeModels()


def test_openai_inventory_fetcher_normalizes_and_sorts_models():
    records = OpenAIInventoryFetcher(client=FakeClient()).fetch()

    assert [record.model_id for record in records] == ["gpt-test-a", "gpt-test-b"]
    assert all(record.provider == "openai" for record in records)
    assert all(record.available for record in records)
    assert records[0].created_at == 1
    assert records[0].owned_by == "openai"
    assert records[0].provenance is not None
    assert records[0].provenance.source == "https://api.openai.com/v1/models"
    assert len(records[0].provenance.content_hash) == 64


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
