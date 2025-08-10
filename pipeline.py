import os
import re
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from io import StringIO
from tqdm import tqdm
from pathlib import Path

def load_scada(root_path, output_dir, spec_path):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = []
    column_names = pd.read_csv(spec_path, header=None).iloc[0].tolist()

    schema_fields = []
    for col in column_names:
        if col.lower() == "date and time":
            schema_fields.append((col, pa.timestamp('ms')))
        else:
            schema_fields.append((col, pa.float64()))
    schema_fields.append(("turbine", pa.string()))
    schema_fields.append(("year", pa.int32()))
    schema = pa.schema(schema_fields)

    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("SCADA_"):
            parts = folder_name.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                year = int(parts[1])
            else:
                continue
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    csv_files.append((os.path.join(folder_path, file_name), year))

    skipped_files = []

    for file_path, year in tqdm(csv_files, desc="Processing SCADA CSVs"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            metadata_lines = [line for line in lines if line.startswith("#")]
            data_start_index = len(metadata_lines)

            turbine_line = next((line for line in metadata_lines if "Turbine:" in line), None)
            if turbine_line:
                cleaned_line = turbine_line.lstrip("#").strip()
                match = re.search(r'Turbine\s*:\s*(.+?)\s+(\d+)', cleaned_line)
                if match:
                    farm_name = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
                    turbine_number = match.group(2)
                    turbine_id = f"turbine_{farm_name}_{turbine_number}"
                else:
                    print(f"[WARN] Could not parse turbine line in: {file_path}")
                    skipped_files.append(file_path)
                    continue
            else:
                print(f"[WARN] Turbine metadata missing in: {file_path}")
                skipped_files.append(file_path)
                continue

            data_str = "".join(lines[data_start_index:])
            df = pd.read_csv(StringIO(data_str), header=None)
            df.columns = column_names
            df['turbine'] = turbine_id
            df['year'] = year
            if "Date and time" in df.columns:
                df["Date and time"] = pd.to_datetime(df["Date and time"], errors="coerce")

            for col in column_names:
                if col.lower() != "date and time":
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            pa_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            ds.write_dataset(
                data=pa_table,
                base_dir=output_dir,
                partitioning=["turbine", "year"],
                format="parquet",
                existing_data_behavior="delete_matching"
            )

        except Exception as e:
            print(f"[ERROR] Failed to process file: {file_path}")
            print(f"        {type(e).__name__}: {e}")
            skipped_files.append(file_path)

    print(f"[INFO] Completed write to: {output_dir}")
    if skipped_files:
        print(f"[INFO] Skipped {len(skipped_files)} files:")
        for path in skipped_files:
            print(f" - {path}")


def load_status(root_path, output_dir, schema=None):
    root_path = Path(root_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_files = []
    for folder in root_path.iterdir():
        if folder.is_dir() and folder.name.startswith("STATUS_"):
            parts = folder.name.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                year = int(parts[1])
            else:
                continue
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() == ".csv":
                    csv_files.append((str(file), year))

    skipped_files = []
    for file_path, year in tqdm(csv_files, desc="Processing STATUS CSVs", unit="file"):
        try:
            # Extract turbine status metadata from commented header lines
            turbine_id = None
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith("#"):
                        break
                    if "Turbine" in line:
                        cleaned = line.lstrip("#").strip()
                        m = re.search(r"Turbine\s*:\s*(.+?)\s+(\d+)", cleaned)
                        if m:
                            farm = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
                            num = m.group(2)
                            turbine_id = f"turbine_{farm}_{num}"
                            break
            if not turbine_id:
                print(f"[WARN] Turbine status metadata missing or unparsable in: {file_path}")
                skipped_files.append(file_path)
                continue

            df = pd.read_csv(file_path, comment="#", header=None, encoding="utf-8")
            df["turbine"] = turbine_id
            df["year"] = year
            df.columns = df.iloc[0]
            df = df[1:]

            pa_table = (
                pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                if schema is not None
                else pa.Table.from_pandas(df, preserve_index=False)
            )

            ds.write_dataset(
                data=pa_table,
                base_dir=output_dir,
                partitioning=["turbine", "year"],
                format="parquet",
                existing_data_behavior="delete_matching",
            )

        except Exception as e:
            print(f"[ERROR] Failed to process file: {file_path}")
            print(f"        {type(e).__name__}: {e}")
            skipped_files.append(file_path)

    print(f"[INFO] Completed write to: {output_dir}")
    if skipped_files:
        print(f"[INFO] Skipped {len(skipped_files)} files:")
        for path in skipped_files:
            print(f" - {path}")


def dataset_spitter(farm_name, data_category, turbine_name=None):
    """
    Returns a dataframe and the valid categories for which we consider the turbine data
    """
    base_path = Path().resolve()
    data_category = str(data_category).upper()

    if data_category == 'SCADA_DATA':
        dataset_path = base_path / farm_name / data_category
        if turbine_name:
            dataset_path = dataset_path / turbine_name
        dataset = ds.dataset(dataset_path, format='parquet')
        return dataset.to_table().to_pandas()

    elif data_category == 'STATUS_DATA':
        dataset_path = base_path / farm_name / data_category
        if turbine_name:
            dataset_path = dataset_path / turbine_name
        dataset = ds.dataset(dataset_path, format='parquet')
        df = dataset.to_table().to_pandas()
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        return df

    else:
        raise ValueError("data_category must be 'SCADA_DATA' or 'STATUS_DATA'.")


def build_clean_events(
    df_status,
    start_col="Timestamp start",
    end_col="Timestamp end",
    dur_col_alt="Duration_hours",     
    dur_text_col="Duration",           
    iec_col="IEC category",
    valid_categories=("Out of Electrical Specification", "Full Performance"),
    min_valid_hours=0,
):
    """
    Returns a DataFrame with exactly:
      start, end, duration_hours, is_valid, is_invalid

    Steps:
      - coerce timestamps
      - derive duration_hours (prefer textual 'Duration' -> timedelta; else 'Duration_hours'; else end-start)
      - synthesize missing end from start + duration_hours
      - drop bad spans
      - compute validity (IEC category in valid_categories AND duration_hours >= min_valid_hours)
      - remove overlapping intervals (keep earliest)
    """
    need = {start_col, iec_col}
    if not need.issubset(df_status.columns):
        missing = ", ".join(sorted(need - set(df_status.columns)))
        raise KeyError(f"df_status missing required columns: {missing}")

    df = df_status.copy()

    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    if end_col in df.columns:
        df[end_col] = pd.to_datetime(df.get(end_col), errors="coerce")
    else:
        df[end_col] = pd.NaT

    if dur_text_col in df.columns:
        td = pd.to_timedelta(df[dur_text_col], errors="coerce")
        df["duration_hours"] = td / pd.Timedelta(hours=1)
    elif dur_col_alt in df.columns:
        df["duration_hours"] = pd.to_numeric(df[dur_col_alt], errors="coerce")
    else:
        df["duration_hours"] = pd.NA

    need_end = df[end_col].isna() & df[start_col].notna() & pd.notna(df["duration_hours"])
    if need_end.any():
        df.loc[need_end, end_col] = df.loc[need_end, start_col] + pd.to_timedelta(
            df.loc[need_end, "duration_hours"], unit="h"
        )
    still_missing = pd.isna(df["duration_hours"]) & df[end_col].notna() & df[start_col].notna()
    if still_missing.any():
        span = (df.loc[still_missing, end_col] - df.loc[still_missing, start_col]).dt.total_seconds() / 3600.0
        df.loc[still_missing, "duration_hours"] = span


    df = df.dropna(subset=[start_col, end_col, "duration_hours"]).copy()
    df = df.loc[df[end_col] > df[start_col]].copy()

    if df.empty:
        return pd.DataFrame(columns=["start", "end", "duration_hours", "is_valid", "is_invalid"])

    valid_set = set(valid_categories)
    s = df[iec_col].astype("string").str.strip()
    df["is_valid"] = s.isin(valid_set) & (df["duration_hours"] >= float(min_valid_hours))
    df["is_valid"] = df["is_valid"].fillna(False)
    df["is_invalid"] = ~df["is_valid"]

    df = df.sort_values(start_col).reset_index(drop=True)
    kept = []
    last_end = None
    for _, r in df.iterrows():
        s_ts, e_ts = r[start_col], r[end_col]
        if (last_end is None) or (s_ts >= last_end):
            kept.append((s_ts, e_ts, float(r["duration_hours"]), bool(r["is_valid"]), bool(r["is_invalid"])))
            last_end = e_ts

    out = pd.DataFrame(kept, columns=["start", "end", "duration_hours", "is_valid", "is_invalid"])
    return out.reset_index(drop=True)


if __name__ == '__main__':
    farm_name = 'penmanshiel'

    df_status = build_clean_events(dataset_spitter(f'{farm_name}', 'STATUS_DATA'))
    df_scada = dataset_spitter(f'{farm_name}', 'SCADA_DATA')
    df_scada = df_scada.rename({'Date and time':'time'}, axis=1)
    time_to_use = df_status[(df_status['duration_hours']>24)&(df_status['is_valid']==True)].drop(columns=['is_valid', 'is_invalid'])
    mask = pd.Series(False, index=df_scada.index)
    for _, row in time_to_use.iterrows():
        mask |= (df_scada['time'] >= row['start']) & (df_scada['time'] <= row['end'])

    df_scada[mask].to_parquet(f"{Path().resolve()}/curated/curated_data_{farm_name}.parquet")

    



