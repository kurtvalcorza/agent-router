from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
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
    if "$" not in cleaned:
        raise PricingSourceError(f"expected USD/MTok price, got {value!r}")
    number = cleaned.split("$", 1)[1].split("/", 1)[0].strip()
    try:
        return float(number)
    except ValueError as exc:
        raise PricingSourceError(f"invalid price {value!r}") from exc


def _input_output_rates(value: str) -> tuple[float, float]:
    matches = re.findall(r"(Input|Output)\s*:\s*\$([0-9.]+)\s*/\s*MTok", value)
    rates = {kind.lower(): float(amount) for kind, amount in matches}
    if "input" not in rates or "output" not in rates:
        raise PricingSourceError(f"expected input/output long-context rates, got {value!r}")
    return rates["input"], rates["output"]


class AnthropicPricingSource:
    """Parse Anthropic's official pricing page using reviewed model mappings."""

    def __init__(
        self,
        model_ids: dict[str, str],
        *,
        long_context_thresholds: dict[str, int] | None = None,
        url: str = ANTHROPIC_PRICING_URL,
        fetch_text: Callable[[str], str] = _default_fetch_text,
        parser_version: str = "anthropic-html-v2",
    ) -> None:
        self.model_ids = dict(model_ids)
        self.long_context_thresholds = dict(long_context_thresholds or {})
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
        long_context = self._find_long_context_table(parser.tables)
        long_rates = self._parse_long_context_table(long_context) if long_context is not None else None
        batch_rows = self._rows_by_model(batch) if batch is not None else {}
        base_rows = self._rows_by_model(base)
        provenance = SourceProvenance.from_payload(
            source=self.url,
            payload=text,
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

            threshold = self.long_context_thresholds.get(display_name)
            if threshold is not None:
                if long_rates is None:
                    raise PricingSourceError(
                        f"long-context pricing requested for {display_name!r}, but rule table was not found"
                    )
                long_input, long_output = long_rates
                pricing = replace(
                    pricing,
                    long_context_input=long_input,
                    long_context_output=long_output,
                    long_context_threshold=threshold,
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
    def _find_long_context_table(tables: list[list[list[str]]]) -> list[list[str]] | None:
        for table in tables:
            if not table or len(table[0]) != 2:
                continue
            left, right = table[0]
            if "200K input tokens" in left and "200K input tokens" in right:
                return table
        return None

    @staticmethod
    def _parse_long_context_table(table: list[list[str]]) -> tuple[float, float]:
        if len(table) < 2 or len(table[1]) != 2:
            raise PricingSourceError("long-context pricing table has unexpected shape")
        standard_input, standard_output = _input_output_rates(table[1][0])
        premium_input, premium_output = _input_output_rates(table[1][1])
        if premium_input < standard_input or premium_output < standard_output:
            raise PricingSourceError("long-context premium rates must not be below standard rates")
        return premium_input, premium_output

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
