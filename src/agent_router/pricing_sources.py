from __future__ import annotations

from collections.abc import Callable
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from .pricing import PricingProfile
from .provenance import PricingRecord, SourceProvenance

ANTHROPIC_PRICING_URL = "https://docs.anthropic.com/en/docs/about-claude/pricing"


class PricingSourceError(RuntimeError):
    pass


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _default_fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "agent-router/0.1 pricing-audit"})
    with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed/explicit HTTPS source
        return response.read().decode("utf-8")


def _usd_per_mtok(value: str) -> float:
    cleaned = value.replace(",", "")
    marker = "$"
    if marker not in cleaned:
        raise PricingSourceError(f"expected USD/MTok price, got {value!r}")
    number = cleaned.split(marker, 1)[1].split("/", 1)[0].strip()
    try:
        return float(number)
    except ValueError as exc:
        raise PricingSourceError(f"invalid price {value!r}") from exc


class AnthropicPricingSource:
    """Parse Anthropic's official pricing table with explicit display-name mapping.

    The source adapter intentionally does not guess API model IDs from marketing names.
    Callers provide a reviewed mapping from pricing-page display names to catalog IDs.
    """

    def __init__(
        self,
        model_ids: dict[str, str],
        *,
        url: str = ANTHROPIC_PRICING_URL,
        fetch_text: Callable[[str], str] = _default_fetch_text,
        parser_version: str = "anthropic-html-v1",
    ) -> None:
        self.model_ids = dict(model_ids)
        self.url = url
        self.fetch_text = fetch_text
        self.parser_version = parser_version

    def fetch(self) -> tuple[PricingRecord, ...]:
        text = self.fetch_text(self.url)
        parser = _TableParser()
        parser.feed(text)
        base = self._find_table(
            parser.tables,
            required={"Model", "Base Input Tokens", "Cache Hits & Refreshes", "Output Tokens"},
        )
        batch = self._find_table(
            parser.tables,
            required={"Model", "Batch input", "Batch output"},
            required_table=False,
        )
        batch_rows = self._rows_by_model(batch) if batch is not None else {}
        base_rows = self._rows_by_model(base)
        provenance = SourceProvenance.from_text(
            source=self.url,
            text=text,
            parser_version=self.parser_version,
        )

        records: list[PricingRecord] = []
        for display_name, model_id in self.model_ids.items():
            row = base_rows.get(display_name)
            if row is None:
                raise PricingSourceError(f"pricing row not found for {display_name!r}")
            batch_row = batch_rows.get(display_name)
            pricing = PricingProfile(
                standard_input=_usd_per_mtok(row["Base Input Tokens"]),
                standard_output=_usd_per_mtok(row["Output Tokens"]),
                cached_input=_usd_per_mtok(row["Cache Hits & Refreshes"]),
                cache_write=_usd_per_mtok(row["5m Cache Writes"])
                if "5m Cache Writes" in row
                else None,
                batch_input=_usd_per_mtok(batch_row["Batch input"])
                if batch_row is not None
                else None,
                batch_output=_usd_per_mtok(batch_row["Batch output"])
                if batch_row is not None
                else None,
            )
            records.append(
                PricingRecord(
                    provider="anthropic",
                    model_id=model_id,
                    pricing=pricing,
                    provenance=provenance,
                    metadata={"pricing_display_name": display_name},
                )
            )
        return tuple(records)

    @staticmethod
    def _find_table(
        tables: list[list[list[str]]],
        *,
        required: set[str],
        required_table: bool = True,
    ) -> list[list[str]] | None:
        for table in tables:
            if table and required.issubset(set(table[0])):
                return table
        if required_table:
            raise PricingSourceError(f"pricing table missing expected columns: {sorted(required)!r}")
        return None

    @staticmethod
    def _rows_by_model(table: list[list[str]] | None) -> dict[str, dict[str, str]]:
        if table is None:
            return {}
        headers = table[0]
        rows: dict[str, dict[str, str]] = {}
        for values in table[1:]:
            if len(values) != len(headers):
                continue
            row = dict(zip(headers, values, strict=True))
            model = row.get("Model")
            if model:
                rows[model] = row
        return rows
