from pathlib import Path
import duckdb
import pandas as pd
import xarray as xr
import yaml


def load_ontology(path: str = "ontology/ontology.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bidding_zone_mart(
    zarr_dir: str = "data/zarr",
    lakehouse_dir: str = "data/lakehouse",
) -> Path:
    ontology = load_ontology()
    zones = ontology["entities"]["BiddingZone"]["members"]
    df_rule = ontology["metrics"]["dunkelflaute_event"]["thresholds"]
    max_cf = float(df_rule["max_capacity_factor"])
    min_hours = int(df_rule["min_consecutive_hours"])

    # Pick latest Zarr v3 store
    zarr_stores = sorted(Path(zarr_dir).glob("*.zarr"))
    if not zarr_stores:
        raise FileNotFoundError(f"No Zarr stores found in {zarr_dir}. Run Step 2 ingestion first.")
    latest_zarr = zarr_stores[-1]

    ds = xr.open_zarr(latest_zarr, consolidated=False)
    run_cycle = ds.attrs.get("run_cycle", "unknown")
    model_id = ds.attrs.get("model_id", "ecmwf_aifs")

    rows: list[dict] = []
    for z in zones:
        # Spatial slice per bidding zone (ECMWF latitudes run North-to-South)
        sub = ds.sel(
            latitude=slice(z["lat_max"], z["lat_min"]),
            longitude=slice(z["lon_min"], z["lon_max"]),
        )
        # Spatial quantiles across grid cells per forecast step
        cf_p10 = sub["wind_capacity_factor"].quantile(0.10, dim=["latitude", "longitude"]).values
        cf_p50 = sub["wind_capacity_factor"].quantile(0.50, dim=["latitude", "longitude"]).values
        cf_p90 = sub["wind_capacity_factor"].quantile(0.90, dim=["latitude", "longitude"]).values
        ws_mean = sub["wind_speed_100m_ms"].mean(dim=["latitude", "longitude"]).values

        steps_td = pd.to_timedelta(sub["step"].values)
        valid_times = pd.to_datetime(sub["valid_time"].values)

        for idx, td in enumerate(steps_td):
            lead_hours = int(td.total_seconds() // 3600)
            p50_val = round(float(cf_p50[idx]), 4)
            rows.append(
                {
                    "run_cycle": run_cycle,
                    "model_id": model_id,
                    "zone_code": z["zone_code"],
                    "zone_name": z["name"],
                    "installed_wind_gw": float(z["installed_wind_gw"]),
                    "lead_hours": lead_hours,
                    "valid_time_utc": valid_times[idx],
                    "wind_speed_mean_ms": round(float(ws_mean[idx]), 2),
                    "cf_p10": round(float(cf_p10[idx]), 4),
                    "cf_p50": p50_val,
                    "cf_p90": round(float(cf_p90[idx]), 4),
                    "expected_gen_gw": round(p50_val * float(z["installed_wind_gw"]), 2),
                    "is_low_wind_step": p50_val < max_cf,
                }
            )

    df = pd.DataFrame(rows).sort_values(["zone_code", "lead_hours"]).reset_index(drop=True)

    # Calculate consecutive low-wind duration per bidding zone (6h ECMWF steps)
    step_hours = 6
    consecutive_hours = []
    running = 0
    prev_zone = None
    for _, r in df.iterrows():
        if r["zone_code"] != prev_zone:
            running = 0
            prev_zone = r["zone_code"]
        running = (running + step_hours) if r["is_low_wind_step"] else 0
        consecutive_hours.append(running)

    df["consecutive_low_wind_hours"] = consecutive_hours
    df["dunkelflaute_alert"] = df["consecutive_low_wind_hours"] >= min_hours

    out_dir = Path(lakehouse_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "gold_bidding_zone_wind_risk.parquet"
    db_path = out_dir / "euro_wind_lakehouse.duckdb"

    df.to_parquet(parquet_path, index=False)

    # Materialize Gold table & semantic view in DuckDB (portable SQL contract)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gold.bidding_zone_wind_risk AS
        SELECT * FROM read_parquet('{parquet_path.as_posix()}');
        """
    )
    summary = con.execute(
        """
        SELECT
            zone_code,
            COUNT(*) AS forecast_steps,
            ROUND(MIN(cf_p50), 3) AS min_cf_p50,
            ROUND(MAX(cf_p50), 3) AS max_cf_p50,
            MAX(consecutive_low_wind_hours) AS max_drought_hrs,
            BOOL_OR(dunkelflaute_alert) AS has_dunkelflaute_alert
        FROM gold.bidding_zone_wind_risk
        GROUP BY zone_code
        ORDER BY min_cf_p50 ASC;
        """
    ).df()
    con.close()

    print(f"Materialized Gold Mart ({len(df)} rows) -> {parquet_path} & {db_path}")
    print(summary.to_string(index=False))
    return db_path


if __name__ == "__main__":
    build_bidding_zone_mart()