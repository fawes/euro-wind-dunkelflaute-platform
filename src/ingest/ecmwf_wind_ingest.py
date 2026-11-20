from pathlib import Path
import numpy as np
import xarray as xr
import yaml
from ecmwf.opendata import Client


def load_ontology(path: str = "ontology/ontology.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_wind_capacity_factor(wind_speed_ms: xr.DataArray) -> xr.DataArray:
    """Standard IEC Class II/III normalized power curve (cut-in 3.0, rated 12.5, cut-out 25.0 m/s)."""
    cut_in, rated, cut_out = 3.0, 12.5, 25.0
    cf = xr.zeros_like(wind_speed_ms, dtype=np.float32)

    ramp_mask = (wind_speed_ms >= cut_in) & (wind_speed_ms < rated)
    rated_mask = (wind_speed_ms >= rated) & (wind_speed_ms <= cut_out)

    cf = xr.where(ramp_mask, ((wind_speed_ms - cut_in) / (rated - cut_in)) ** 3, cf)
    cf = xr.where(rated_mask, 1.0, cf)
    cf.name = "wind_capacity_factor"
    cf.attrs = {"units": "ratio", "valid_min": 0.0, "valid_max": 1.0}
    return cf


def ingest_ecmwf_aifs_wind(
    steps: list[int] | None = None,
    output_dir: str = "data",
) -> Path:
    ontology = load_ontology()
    zones = ontology["entities"]["BiddingZone"]["members"]

    # Derive tight European bounding box from ontology bidding zones + 1-deg buffer
    lat_min = min(z["lat_min"] for z in zones) - 1.0
    lat_max = max(z["lat_max"] for z in zones) + 1.0
    lon_min = min(z["lon_min"] for z in zones) - 1.0
    lon_max = max(z["lon_max"] for z in zones) + 1.0

    steps = steps or [0, 6, 12, 18, 24, 30, 36, 42, 48]
    raw_dir = Path(output_dir) / "raw"
    zarr_dir = Path(output_dir) / "zarr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zarr_dir.mkdir(parents=True, exist_ok=True)

    grib_path = raw_dir / "ecmwf_aifs_100m_wind.grib2"

    # 1. Fetch 100u and 100v from ECMWF Open Data (AIFS deterministic model)
    client = Client(source="ecmwf", model="aifs-single", resol="0p25")
    result = client.retrieve(
        type="fc",
        param=["100u", "100v"],
        step=steps,
        target=str(grib_path),
    )
    run_tag = result.datetime.strftime("%Y%m%d_%Hz")

    # 2. Decode GRIB2 -> Xarray & normalize longitude (-180..180)
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    if float(ds.longitude.max()) > 180.0:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")

    # 3. Spatial domain pruning in RAM (ECMWF latitudes run North-to-South)
    ds_eu = ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

    # 4. Derive wind speed magnitude & capacity factor
    wind_speed = np.hypot(ds_eu["u100"], ds_eu["v100"]).astype(np.float32)
    wind_speed.name = "wind_speed_100m_ms"
    wind_speed.attrs = {"units": "m/s", "long_name": "100m Hub-Height Wind Speed"}

    capacity_factor = compute_wind_capacity_factor(wind_speed)

    out_ds = xr.Dataset(
        {
            "u100": ds_eu["u100"].astype(np.float32),
            "v100": ds_eu["v100"].astype(np.float32),
            "wind_speed_100m_ms": wind_speed,
            "wind_capacity_factor": capacity_factor,
        },
        attrs={
            "model_id": "ecmwf_aifs",
            "ontology_version": ontology["version"],
            "run_cycle": run_tag,
        },
    )

    # 5. Persist chunked Zarr v3 store
    zarr_path = zarr_dir / f"ecmwf_aifs_{run_tag}.zarr"
    out_ds.chunk({"step": 3, "latitude": 64, "longitude": 64}).to_zarr(
        zarr_path,
        mode="w",
        zarr_format=3,
    )

    raw_mb = grib_path.stat().st_size / (1024 * 1024)
    print(f"Run {run_tag} | Global GRIB2: {raw_mb:.2f} MB -> Pruned EU Zarr v3: {zarr_path}")
    print(out_ds)
    return zarr_path


if __name__ == "__main__":
    ingest_ecmwf_aifs_wind()