import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOTS_PATH = ROOT / "src" / "data" / "tenders" / "processed" / "lots.jsonl"
OUTPUT_PATH = ROOT / "db" / "init" / "10_seed.sql"


def sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    text = text.replace("\\", "\\\\").replace("'", "''")
    text = text.replace("\r", "").replace("\n", "\\n")
    return f"E'{text}'"


def _short_title(value: str | None, max_len: int = 160) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    trimmed = text[:max_len].rsplit(" ", 1)[0]
    if not trimmed:
        trimmed = text[:max_len]
    return f"{trimmed}..."


def main():
    tenders = {}

    with LOTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tender_id = Path(data.get("source", "")).stem
            if not tender_id:
                continue
            entry = tenders.get(tender_id)
            title = _short_title(data.get("title"))
            if entry is None:
                tenders[tender_id] = {
                    "tender_id": tender_id,
                    "source_file": data.get("source"),
                    "primary_title": title,
                    "summary": None,
                    "notes": "Seeded from tender metadata only (no PDF content stored).",
                }
                continue
            if not entry["primary_title"] and title:
                entry["primary_title"] = title

    lines = []
    lines.append("BEGIN;")
    lines.append("-- Company profile (synthetic demo data)")
    lines.append(
        "WITH new_profile AS ("
        "INSERT INTO company_profile "
        "(name, description, country, website, notes, source_notes) VALUES "
        "('Baltic Industrial Works Ltd.', "
        "'Fabrication and machining of steel components for power and industrial plants.', "
        "'Poland', "
        "'https://example.com', "
        "'Synthetic demo profile for readiness checks.', "
        "'Synthetic data; replace with real company profile.') "
        "RETURNING id"
        ") "
        "INSERT INTO company_capability "
        "(profile_id, capability, capacity, certification, source_notes) "
        "SELECT new_profile.id, t.capability, t.capacity, t.certification, t.source_notes "
        "FROM new_profile "
        "CROSS JOIN (VALUES "
        "('Pressure parts fabrication (boilers, superheaters)', "
        "'Up to 12,000 kg assemblies; 45 t/month', "
        "'UDT approval; ISO 3834-2', "
        "'Synthetic demo data'), "
        "('Mechanical drives (gearboxes, reducers)', "
        "'Custom and batch production', "
        "'ISO 9001', "
        "'Synthetic demo data'), "
        "('Field installation and commissioning', "
        "'On-site teams across Poland', "
        "'Welding EN 287-1', "
        "'Synthetic demo data')"
        ") AS t(capability, capacity, certification, source_notes);"
    )
    lines.append("")
    lines.append("-- Tenders")
    for tender in tenders.values():
        lines.append(
            "INSERT INTO tenders (tender_id, source_file, primary_title, summary, notes) VALUES "
            f"({sql_literal(tender['tender_id'])}, {sql_literal(tender['source_file'])}, "
            f"{sql_literal(tender['primary_title'])}, {sql_literal(tender['summary'])}, "
            f"{sql_literal(tender['notes'])});"
        )
    lines.append("")
    lines.append("-- Company delivery history (synthetic demo data)")
    deliveries = [
        {
            "tender_id": "ted_5789-2018_EN",
            "title": "Gearbox assemblies for conveyor drives",
            "customer": "PGG S.A. (synthetic)",
            "delivered_at": "2023-05-20",
            "value": 450000,
            "currency": "PLN",
            "scope": "Production and delivery of industrial gearboxes for mining conveyors.",
            "evidence": "Reference letter (synthetic)",
            "source_notes": "Synthetic demo data",
        },
        {
            "tender_id": "ted_5789-2018_EN",
            "title": "Motoreducers for heavy-duty handling line",
            "customer": "Industrial client (synthetic)",
            "delivered_at": "2022-09-10",
            "value": 320000,
            "currency": "PLN",
            "scope": "Batch delivery of motoreducers with custom ratios.",
            "evidence": "Delivery acceptance protocol (synthetic)",
            "source_notes": "Synthetic demo data",
        },
        {
            "tender_id": "ted_812-2018_EN",
            "title": "Boiler superheater pressure parts",
            "customer": "Power plant operator (synthetic)",
            "delivered_at": "2021-03-15",
            "value": 680000,
            "currency": "PLN",
            "scope": "Fabrication of pressure parts for OP 230 boilers.",
            "evidence": "Contract and acceptance note (synthetic)",
            "source_notes": "Synthetic demo data",
        },
    ]
    for delivery in deliveries:
        lines.append(
            "INSERT INTO company_delivery "
            "(tender_id, title, customer, delivered_at, value, currency, scope, evidence, source_notes) VALUES "
            f"({sql_literal(delivery['tender_id'])}, {sql_literal(delivery['title'])}, "
            f"{sql_literal(delivery['customer'])}, {sql_literal(delivery['delivered_at'])}, "
            f"{sql_literal(delivery['value'])}, {sql_literal(delivery['currency'])}, "
            f"{sql_literal(delivery['scope'])}, {sql_literal(delivery['evidence'])}, "
            f"{sql_literal(delivery['source_notes'])});"
        )
    lines.append("COMMIT;")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
