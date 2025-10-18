"""Parsing utilities for the ABMAP supplementary datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ColumnMapping:
    heavy: str
    light: Optional[str]
    antigen: str
    target: str
    metadata: List[str]


class ColumnInferenceError(RuntimeError):
    """Raised when the parser cannot infer the schema of a supplementary table."""


def _is_sequence_series(series: pd.Series) -> bool:
    if series.dtype.kind not in {"O", "U", "S"}:
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    for value in sample:
        value = value.strip().upper()
        if not value:
            continue
        if not set(value) <= AMINO_ACIDS:
            return False
    return True


def _is_numeric_series(series: pd.Series) -> bool:
    return series.dtype.kind in {"i", "u", "f"}


def infer_column_mapping(frame: pd.DataFrame) -> ColumnMapping:
    candidates = {column.lower(): column for column in frame.columns}

    def pick(keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            for column_lower, original in candidates.items():
                if key in column_lower:
                    return original
        return None

    heavy = pick(["heavy", "vh", "hc_sequence", "vh_sequence"])
    light = pick(["light", "vl", "lc_sequence", "vl_sequence"])
    antigen = pick(["antigen", "peptide", "epitope", "target_sequence"])
    target = pick(["binding", "affinity", "kd", "score", "signal", "logfc"])

    sequence_columns = [column for column in frame.columns if _is_sequence_series(frame[column])]
    numeric_columns = [column for column in frame.columns if _is_numeric_series(frame[column])]

    if heavy is None and sequence_columns:
        heavy = sequence_columns[0]
    if light is None and len(sequence_columns) > 1:
        for column in sequence_columns:
            if column != heavy:
                light = column
                break
    if antigen is None and sequence_columns:
        antigen = sequence_columns[-1]
    if target is None and numeric_columns:
        target = numeric_columns[0]

    if heavy is None or antigen is None or target is None:
        raise ColumnInferenceError(
            "Unable to infer the heavy chain, antigen, or target columns from supplementary table."
        )

    metadata = [
        column
        for column in frame.columns
        if column not in {heavy, light, antigen, target}
    ]

    return ColumnMapping(heavy=heavy, light=light, antigen=antigen, target=target, metadata=metadata)


def normalize_sequences(frame: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    result = frame.copy()
    for column in filter(None, [mapping.heavy, mapping.light, mapping.antigen]):
        result[column] = (
            result[column]
            .astype(str)
            .str.upper()
            .str.replace("[^A-Z]", "", regex=True)
        )
    return result


def extract_training_records(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = infer_column_mapping(frame)
    normalized = normalize_sequences(frame, mapping)
    columns = [mapping.heavy]
    if mapping.light is not None:
        columns.append(mapping.light)
    columns.extend([mapping.antigen, mapping.target])
    columns.extend(mapping.metadata)
    rename_map = {
        mapping.heavy: "heavy_chain",
        mapping.antigen: "antigen_sequence",
        mapping.target: "binding_score",
    }
    if mapping.light is not None:
        rename_map[mapping.light] = "light_chain"
    return normalized[columns].rename(columns=rename_map)


def load_supplementary_tables(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            excel = pd.ExcelFile(path)
            for sheet in excel.sheet_names:
                frame = excel.parse(sheet)
                if frame.empty:
                    continue
                try:
                    frames.append(extract_training_records(frame))
                except ColumnInferenceError:
                    continue
        elif path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            sep = "," if path.suffix.lower() == ".csv" else "\t"
            frame = pd.read_csv(path, sep=sep)
            if frame.empty:
                continue
            frames.append(extract_training_records(frame))
        else:
            raise ValueError(f"Unsupported supplementary file format: {path.suffix}")
    if not frames:
        raise ColumnInferenceError("No supplementary tables contained recognizable training data.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["heavy_chain", "antigen_sequence", "binding_score"])
    combined = combined.reset_index(drop=True)
    return combined


def save_processed_dataset(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert ABMAP supplementary tables into a training parquet file.")
    parser.add_argument("input", type=Path, nargs="+", help="One or more raw supplementary files.")
    parser.add_argument("--output", type=Path, required=True, help="Destination Parquet file.")
    args = parser.parse_args(argv)

    frame = load_supplementary_tables(args.input)
    save_processed_dataset(frame, args.output)

    print(json.dumps({"records": len(frame), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
