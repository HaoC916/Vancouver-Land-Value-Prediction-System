from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - 64)
    return max(value - 1, 0)


def read_xlsx_sheet_rows(path: Path, sheet_name: str) -> list[list[str]]:
    """Read one XLSX sheet as rows of raw string values without openpyxl."""
    with zipfile.ZipFile(path, "r") as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{NS_PKG}}}Relationship")
        }

        target = None
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib.get(f"{{{NS_REL}}}id")
                if rid:
                    target = "xl/" + rid_to_target[rid]
                break
        if target is None:
            raise ValueError(f"Sheet not found: {sheet_name}")

        root = ET.fromstring(zf.read(target))
        rows: list[list[str]] = []
        for row in root.findall(f".//{{{NS_MAIN}}}row"):
            cells: dict[int, str] = {}
            for cell in row.findall(f"{{{NS_MAIN}}}c"):
                idx = _col_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    text_node = cell.find(f"{{{NS_MAIN}}}is/{{{NS_MAIN}}}t")
                    value = "" if text_node is None else (text_node.text or "")
                else:
                    value_node = cell.find(f"{{{NS_MAIN}}}v")
                    value = "" if value_node is None else (value_node.text or "")
                cells[idx] = value

            if cells:
                max_idx = max(cells)
                rows.append([cells.get(i, "") for i in range(max_idx + 1)])
        return rows


def detect_year_total_columns(rows: list[list[str]]) -> dict[int, int]:
    """Detect columns like '2015 Total', returning {year: column_index}."""
    pattern = re.compile(r"^(20\d{2})\s*Total$")
    best: dict[int, int] = {}
    for row in rows[:15]:
        current = {}
        for idx, value in enumerate(row):
            match = pattern.match(str(value).strip())
            if match:
                current[int(match.group(1))] = idx
        if len(current) > len(best):
            best = current

    if not best:
        raise ValueError("Could not detect 'YYYY Total' columns in XLSX sheet.")
    return best


def parse_number(value: str) -> float:
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "..", "..."}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")
