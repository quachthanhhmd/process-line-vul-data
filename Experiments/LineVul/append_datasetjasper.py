#!/usr/bin/env python3
"""
Real_Vul_data.csv에서 project가 'jasper'인 행을 찾으면 즉시
unique_id로 소스코드를 매핑하여 jasper_data_append_processed_func.csv를 생성합니다.

추가 기능:
- --path 인자에 따라 Dataset / Dataset/test 경로를 선택
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, Tuple

EMPTY_PLACEHOLDER = " "

def resolve_paths(path_arg: str) -> Tuple[Path, Path, Path]:
    """
    --path 인자에 따라 real_vul_csv, source_dir, output_csv를 결정.
    path_arg는 'Dataset' 또는 'Dataset/test' 형태를 기대.
    """
    # 입력을 통일 (앞/뒤 슬래시 제거)
    normalized = path_arg.strip().strip("/")

    if normalized == "Dataset":
        real_vul_csv = Path("/app/RealVul/Dataset/Real_Vul_data.csv")
        source_dir   = Path("/app/RealVul/Dataset/all_source_code")
        output_csv   = Path("/app/RealVul/Dataset/jasper_data_append_processed_func.csv")
        return real_vul_csv, source_dir, output_csv

    if normalized == "Dataset/test":
        real_vul_csv = Path("/app/RealVul/Dataset/test/jasper_dataset.csv")
        source_dir   = Path("/app/RealVul/Dataset/test/source_code")
        output_csv   = Path("/app/RealVul/Dataset/test/jasper_data_append_processed_func.csv")
        return real_vul_csv, source_dir, output_csv

    raise ValueError(
        f"Unsupported --path value: {path_arg}\n"
        "Use one of: 'Dataset', 'Dataset/test'"
    )

def load_source_mapping(source_root: Path) -> Dict[str, Path]:
    """unique_id(확장자 제외) -> 실제 파일 Path 매핑."""
    mapping: Dict[str, Path] = {}
    for path in source_root.iterdir():
        if path.is_file():
            mapping[path.stem] = path
    return mapping

def read_source_text(path: Path) -> str:
    """UTF-8 → ISO-8859-1 순으로 읽기."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="Dataset",
        help="Dataset 경로 선택: 'Dataset' 또는 'Dataset/test' (default: Dataset)",
    )
    args = parser.parse_args()

    real_vul_csv, source_dir, output_csv = resolve_paths(args.path)

    print("Resolved paths:")
    print(f"  - real_vul_csv: {real_vul_csv}")
    print(f"  - source_dir  : {source_dir}")
    print(f"  - output_csv  : {output_csv}")

    print("\nLoading source code mapping...")
    mapping = load_source_mapping(source_dir)
    print(f"  - Indexed {len(mapping)} source files")

    print("\nProcessing CSV (filtering jasper & merging source)...")

    processed = 0
    missing = 0

    with real_vul_csv.open(newline="", encoding="utf-8") as input_fp, \
         output_csv.open("w", newline="", encoding="utf-8") as output_fp:

        reader = csv.DictReader(input_fp)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header (fieldnames).")

        new_fieldnames = list(reader.fieldnames) + ["processed_func"]
        writer = csv.DictWriter(output_fp, fieldnames=new_fieldnames)
        writer.writeheader()

        for row in reader:
            # jasper가 아니면 스킵
            if row.get("project", "").lower() != "jasper":
                continue

            # 즉시 소스코드 매핑
            unique_id = str(row.get("unique_id", ""))
            path = mapping.get(unique_id)

            if path is None:
                missing += 1
                row["processed_func"] = EMPTY_PLACEHOLDER
            else:
                row["processed_func"] = read_source_text(path) or EMPTY_PLACEHOLDER
                processed += 1

            writer.writerow(row)

    print(f"\nCompleted:")
    print(f"  - Jasper rows processed: {processed + missing}")
    print(f"  - Source merged: {processed}")
    print(f"  - Missing source: {missing}")
    print(f"  - Output: {output_csv}")

if __name__ == "__main__":
    main()
