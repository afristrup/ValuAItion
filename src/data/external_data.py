"""
Calculate road distances from apartments to schools using OSRM API.

This module parses school data from InstReg CSV and calculates driving distances
from all apartments to all schools using the OSRM routing service.
"""

import os
import json
import time
import logging
import gc
import pandas as pd
import numpy as np
import requests
from typing import Tuple, Optional, Iterator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from geopy import distance as geopy_distance

    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("geopy not available - geodesic distance calculations disabled")


def parse_schools_data(csv_path: str) -> pd.DataFrame:
    """
    Parse the school CSV file and extract relevant columns.

    Args:
        csv_path: Path to the InstReg CSV file

    Returns:
        DataFrame with school data including coordinates
    """
    logger.info(f"Loading schools data from {csv_path}")

    # Try different encodings (file appears to be UTF-16-LE based on byte pattern)
    encodings = ["utf-16-le", "utf-16", "utf-8", "latin-1", "iso-8859-1", "cp1252"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=encoding)
            # Check if we got valid column names (not just 'H' and 'Unnamed')
            if len(df.columns) > 1 and not all(
                col.startswith("Unnamed") or col == "H" for col in df.columns[:5]
            ):
                logger.info(f"Successfully read CSV with {encoding} encoding")
                break
            else:
                df = None  # Invalid read, try next encoding
        except (UnicodeDecodeError, UnicodeError) as e:
            logger.debug(f"Failed to read with {encoding}: {e}")
            continue
        except Exception as e:
            logger.debug(f"Error reading with {encoding}: {e}")
            continue

    if df is None:
        raise ValueError(
            f"Could not read CSV file with any of the tried encodings: {encodings}"
        )

    # Strip whitespace from column names (they may have leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Select relevant columns (handle potential whitespace in column names)
    required_cols = ["INST_NR", "INST_NAVN", "GEO_BREDDE_GRAD", "GEO_LAENGDE_GRAD"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")
        raise KeyError(f"Required columns not found: {missing_cols}")

    schools_df = df[required_cols].copy()

    # Rename columns for clarity
    schools_df.columns = ["school_id", "school_name", "lat", "lon"]

    # Convert coordinates from comma decimal separator to dot (memory-efficient)
    # Use vectorized operations instead of string operations where possible
    for col in ["lat", "lon"]:
        if schools_df[col].dtype == "object":
            # Only do string replacement if needed (comma-separated decimals)
            schools_df[col] = (
                schools_df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(np.float32)
            )
        else:
            schools_df[col] = schools_df[col].astype(np.float32)

    # Optimize school_id to int32 if possible
    if schools_df["school_id"].dtype == "int64":
        schools_df["school_id"] = schools_df["school_id"].astype(np.int32)

    # Remove rows with missing coordinates
    schools_df = schools_df.dropna(subset=["lat", "lon"])

    logger.info(f"Loaded {len(schools_df)} schools with valid coordinates")

    return schools_df


def get_road_distance(
    point1: dict, point2: dict, max_retries: int = 3, retry_delay: float = 1.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get driving distance and duration between two points using OSRM API.

    Uses the public OSRM routing service (http://project-osrm.org).
    Please use responsibly - for large-scale applications, consider running
    your own OSRM instance or using a commercial service.

    Args:
        point1: Dict with 'lat' and 'lon' keys
        point2: Dict with 'lat' and 'lon' keys
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (distance_meters, duration_seconds) or (None, None) if failed
    """
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{point1['lon']},{point1['lat']};{point2['lon']},{point2['lat']}"
        f"?overview=false&alternatives=false"
    )

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=10)

            if r.status_code == 200:
                response = json.loads(r.content)

                if response.get("code") == "Ok" and response.get("routes"):
                    route = response["routes"][0]
                    distance = route.get("distance")  # in meters
                    duration = route.get("duration")  # in seconds
                    return distance, duration
                else:
                    error_msg = response.get("message", "Unknown error")
                    logger.warning(
                        f"OSRM API returned non-OK code: {response.get('code')} - {error_msg}"
                    )
                    return None, None
            elif r.status_code == 429:
                # Rate limited - wait longer
                logger.warning(
                    f"Rate limited by OSRM API, waiting {retry_delay * 2} seconds..."
                )
                time.sleep(retry_delay * 2)
                continue
            else:
                logger.warning(
                    f"OSRM API returned status code {r.status_code}, attempt {attempt + 1}/{max_retries}"
                )

        except Exception as e:
            logger.warning(
                f"Error calling OSRM API (attempt {attempt + 1}/{max_retries}): {e}"
            )

        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    return None, None


def get_geodesic_distance(point1: dict, point2: dict) -> float:
    """
    Get geodesic (straight-line) distance between two points using geopy.

    This is faster than road distance but less accurate for actual travel.
    Useful as a fallback or for filtering before calculating road distances.

    Args:
        point1: Dict with 'lat' and 'lon' keys
        point2: Dict with 'lat' and 'lon' keys

    Returns:
        Distance in meters
    """
    if not GEOPY_AVAILABLE:
        raise ImportError("geopy is required for geodesic distance calculations")

    d = geopy_distance.distance(
        (point1["lat"], point1["lon"]), (point2["lat"], point2["lon"])
    )
    return d.meters


# Global semaphore for rate limiting (optional, can be None)
_rate_limiter = None


def _get_grid_cell(
    lat: float, lon: float, grid_size_km: float = 10.0
) -> Tuple[int, int]:
    """
    Convert lat/lon coordinates to grid cell indices.

    Args:
        lat: Latitude
        lon: Longitude
        grid_size_km: Size of grid cells in kilometers (default: 10km)

    Returns:
        Tuple of (grid_x, grid_y) cell indices
    """
    # Approximate conversion: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
    # For Denmark (avg lat ~56°), 1 degree lon ≈ 62 km
    # Use average for simplicity
    DEG_LAT_KM = 111.0
    DEG_LON_KM = 62.0  # Approximate for Denmark (latitude ~56°)

    # Denmark bounding box (approximate)
    MIN_LAT = 54.5
    MAX_LAT = 57.8
    MIN_LON = 8.0
    MAX_LON = 15.0

    # Calculate grid cell size in degrees
    grid_lat_deg = grid_size_km / DEG_LAT_KM
    grid_lon_deg = grid_size_km / DEG_LON_KM

    # Calculate grid indices
    grid_x = int((lon - MIN_LON) / grid_lon_deg)
    grid_y = int((lat - MIN_LAT) / grid_lat_deg)

    return (grid_x, grid_y)


def _get_neighboring_cells(
    grid_x: int, grid_y: int, max_distance_cells: int = 1
) -> set:
    """
    Get neighboring grid cells including the cell itself.

    Args:
        grid_x: X grid coordinate
        grid_y: Y grid coordinate
        max_distance_cells: Maximum distance in cells to consider (1 = same + adjacent cells)

    Returns:
        Set of (grid_x, grid_y) tuples for neighboring cells
    """
    neighbors = set()
    for dx in range(-max_distance_cells, max_distance_cells + 1):
        for dy in range(-max_distance_cells, max_distance_cells + 1):
            neighbors.add((grid_x + dx, grid_y + dy))
    return neighbors


def _build_spatial_index(
    lats: np.ndarray,
    lons: np.ndarray,
    grid_size_km: float = 10.0,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Build a spatial index mapping grid cells to point indices.

    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        grid_size_km: Size of grid cells in kilometers

    Returns:
        Tuple of (grid_to_indices dict, grid_x array, grid_y array)
    """
    grid_to_indices = {}
    grid_x_arr = np.zeros(len(lats), dtype=np.int32)
    grid_y_arr = np.zeros(len(lats), dtype=np.int32)

    for idx in range(len(lats)):
        grid_x, grid_y = _get_grid_cell(
            float(lats[idx]), float(lons[idx]), grid_size_km
        )
        grid_x_arr[idx] = grid_x
        grid_y_arr[idx] = grid_y

        cell_key = (grid_x, grid_y)
        if cell_key not in grid_to_indices:
            grid_to_indices[cell_key] = []
        grid_to_indices[cell_key].append(idx)

    return grid_to_indices, grid_x_arr, grid_y_arr


def _calculate_single_distance(
    apartment_id,
    apt_point: dict,
    school_id: int,
    school_name: str,
    school_point: dict,
    use_geodesic: bool = False,
) -> dict:
    """
    Calculate distance for a single apartment-school pair.
    Helper function for multithreading.

    Args:
        apartment_id: ID of the apartment
        apt_point: Dict with 'lat' and 'lon' keys for apartment
        school_id: ID of the school
        school_name: Name of the school
        school_point: Dict with 'lat' and 'lon' keys for school
        use_geodesic: If True, use geodesic (straight-line) distance instead of road distance

    Returns:
        Dict with distance information
    """
    if use_geodesic:
        # Use geodesic distance (straight-line, no API call needed)
        if not GEOPY_AVAILABLE:
            logger.error("geopy not available for geodesic distance calculation")
            distance_m, duration_s = None, None
        else:
            distance_m = get_geodesic_distance(apt_point, school_point)
            duration_s = None  # No duration for straight-line distance
    else:
        # Use road distance (OSRM API)
        # Use rate limiter if set
        if _rate_limiter is not None:
            _rate_limiter.acquire()
            try:
                distance_m, duration_s = get_road_distance(apt_point, school_point)
            finally:
                _rate_limiter.release()
        else:
            distance_m, duration_s = get_road_distance(apt_point, school_point)

    return {
        "TRANSACTION_ID": apartment_id,
        "school_id": school_id,
        "school_name": school_name,
        "distance_m": distance_m,
        "duration_s": duration_s,
    }


def calculate_apartment_school_distances(
    apartments_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    output_path: str,
    max_workers: int = 10,
    batch_size: int = 1000,
    rate_limit: Optional[int] = None,
    use_geodesic: bool = False,
    grid_size_km: float = 10.0,
    max_distance_km: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate distances from all apartments to all schools using multithreading.
    Uses spatial grid filtering to only calculate distances for nearby pairs.

    Args:
        apartments_df: DataFrame with apartment data (must have LAT, LNG, TRANSACTION_ID)
        schools_df: DataFrame with school data (must have lat, lon, school_id)
        output_path: Path to save the results CSV
        max_workers: Maximum number of worker threads for parallel processing
        batch_size: Number of results to collect before saving progress (for large datasets)
        rate_limit: Optional rate limit (max concurrent requests). If None, uses max_workers.
        use_geodesic: If True, use geodesic (straight-line) distance instead of road distance.
                     Much faster but less accurate for actual travel.
        grid_size_km: Size of grid cells in kilometers for spatial filtering (default: 10km)
        max_distance_km: Maximum distance to consider (None = no limit, uses grid filtering only)

    Returns:
        DataFrame with distances (TRANSACTION_ID, school_id, distance_m, duration_s)
    """
    global _rate_limiter

    distance_type = "geodesic (straight-line)" if use_geodesic else "road"

    if use_geodesic:
        if not GEOPY_AVAILABLE:
            raise ImportError(
                "geopy is required for geodesic distance calculations. "
                "Install it with: pip install geopy"
            )
        # No rate limiting needed for geodesic (no API calls)
        # Can use more workers since it's just computation
        _rate_limiter = None
        # Increase workers for geodesic if using default (no API rate limits to worry about)
        if max_workers == 10:  # Default value, increase for geodesic
            estimated_tasks = len(apartments_df) * len(schools_df)
            max_workers = min(50, max(20, estimated_tasks // 1000 + 10))
            logger.info(
                f"Auto-adjusted to {max_workers} workers for geodesic distance "
                f"(no API rate limits)"
            )
    else:
        # Set up rate limiting if specified (only for road distance)
        if rate_limit is not None:
            _rate_limiter = Semaphore(rate_limit)
            logger.info(f"Rate limiting enabled: max {rate_limit} concurrent requests")
        else:
            _rate_limiter = None

    logger.info(
        f"Calculating {distance_type} distances for {len(apartments_df)} apartments "
        f"to {len(schools_df)} schools using {max_workers} threads"
    )

    # Filter apartments with valid coordinates upfront (use view, not copy)
    valid_mask = apartments_df[["LAT", "LNG"]].notna().all(axis=1)
    valid_apartments = apartments_df[valid_mask]
    if len(valid_apartments) < len(apartments_df):
        skipped = len(apartments_df) - len(valid_apartments)
        logger.warning(f"Skipping {skipped} apartments with missing coordinates")

    # Pre-extract school data as numpy arrays for faster access (use memory-efficient types)
    school_ids = schools_df["school_id"].values
    school_names = schools_df["school_name"].values
    school_lats = schools_df["lat"].values.astype(np.float32)
    school_lons = schools_df["lon"].values.astype(np.float32)

    # Pre-extract apartment data - handle TRANSACTION_ID column if it exists
    if "TRANSACTION_ID" in valid_apartments.columns:
        apt_ids = valid_apartments["TRANSACTION_ID"].values
    else:
        apt_ids = valid_apartments.index.values
    apt_lats = valid_apartments["LAT"].values.astype(np.float32)
    apt_lons = valid_apartments["LNG"].values.astype(np.float32)

    # Build spatial grid indices for filtering
    logger.info(f"Building spatial grid index (grid size: {grid_size_km}km)...")
    school_grid_index, school_grid_x, school_grid_y = _build_spatial_index(
        school_lats, school_lons, grid_size_km
    )
    apt_grid_x = np.zeros(len(apt_lats), dtype=np.int32)
    apt_grid_y = np.zeros(len(apt_lats), dtype=np.int32)
    for idx in range(len(apt_lats)):
        apt_grid_x[idx], apt_grid_y[idx] = _get_grid_cell(
            float(apt_lats[idx]), float(apt_lons[idx]), grid_size_km
        )

    # Calculate total combinations with grid filtering
    total_without_grid = len(valid_apartments) * len(schools_df)

    # Count pairs that will be processed (in same or adjacent grid cells)
    total_combinations = 0
    apt_cells_processed = set()

    logger.info("Counting pairs after grid filtering...")
    for apt_idx in range(len(valid_apartments)):
        apt_grid_cell = (apt_grid_x[apt_idx], apt_grid_y[apt_idx])
        if apt_grid_cell not in apt_cells_processed:
            apt_cells_processed.add(apt_grid_cell)

        # Get neighboring cells (including current cell)
        neighbor_cells = _get_neighboring_cells(
            apt_grid_x[apt_idx], apt_grid_y[apt_idx]
        )

        # Count schools in neighboring cells
        for cell in neighbor_cells:
            if cell in school_grid_index:
                total_combinations += len(school_grid_index[cell])

    reduction = (
        (1 - total_combinations / total_without_grid) * 100
        if total_without_grid > 0
        else 0
    )
    logger.info(
        f"Grid filtering: {total_combinations:,} pairs to process "
        f"(reduced from {total_without_grid:,} by {reduction:.1f}%)"
    )

    # Generator function to yield tasks on-demand (memory efficient, with grid filtering)
    def task_generator() -> Iterator[tuple]:
        """Generate tasks on-demand, only for pairs in same/adjacent grid cells."""
        for apt_idx in range(len(valid_apartments)):
            apt_point = {
                "lat": float(apt_lats[apt_idx]),
                "lon": float(apt_lons[apt_idx]),
            }

            # Get neighboring grid cells for this apartment
            neighbor_cells = _get_neighboring_cells(
                apt_grid_x[apt_idx], apt_grid_y[apt_idx]
            )

            # Only process schools in neighboring cells
            for cell in neighbor_cells:
                if cell in school_grid_index:
                    for school_idx in school_grid_index[cell]:
                        # Optional: additional distance check if max_distance_km is set
                        if max_distance_km is not None:
                            if GEOPY_AVAILABLE:
                                # Quick geodesic check (works for both geodesic and road distance)
                                try:
                                    dist_km = (
                                        get_geodesic_distance(
                                            apt_point,
                                            {
                                                "lat": float(school_lats[school_idx]),
                                                "lon": float(school_lons[school_idx]),
                                            },
                                        )
                                        / 1000.0
                                    )
                                    if dist_km > max_distance_km:
                                        continue
                                except Exception:
                                    # If distance check fails, include the pair anyway
                                    pass

                        yield (
                            apt_ids[apt_idx],
                            apt_point,
                            school_ids[school_idx],
                            school_names[school_idx],
                            {
                                "lat": float(school_lats[school_idx]),
                                "lon": float(school_lons[school_idx]),
                            },
                        )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process tasks in parallel with progress bar, appending to CSV in batches
    results_buffer = []
    results_lock = Lock()
    results_count = 0
    file_initialized = False

    def process_with_progress(task):
        """Wrapper to process task and update progress."""
        # Add use_geodesic parameter to task
        result = _calculate_single_distance(*task, use_geodesic=use_geodesic)
        return result

    def flush_results_to_csv():
        """Flush buffered results to CSV file (memory-efficient)."""
        nonlocal file_initialized
        if not results_buffer:
            return

        # Convert buffer to DataFrame with optimized dtypes
        batch_df = pd.DataFrame(results_buffer)

        # Optimize data types to reduce memory
        if "distance_m" in batch_df.columns:
            batch_df["distance_m"] = pd.to_numeric(
                batch_df["distance_m"], errors="coerce", downcast="float"
            )
        if "duration_s" in batch_df.columns:
            batch_df["duration_s"] = pd.to_numeric(
                batch_df["duration_s"], errors="coerce", downcast="float"
            )
        if "school_id" in batch_df.columns:
            batch_df["school_id"] = pd.to_numeric(
                batch_df["school_id"], errors="coerce", downcast="integer"
            )

        # Write to CSV (append mode after first write)
        if not file_initialized:
            batch_df.to_csv(output_path, index=False, mode="w")
            file_initialized = True
        else:
            batch_df.to_csv(output_path, index=False, mode="a", header=False)

        # Clear buffer and force garbage collection
        results_buffer.clear()
        del batch_df
        gc.collect()

    # Use generator and submit tasks in batches to control memory
    task_gen = task_generator()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks in batches to avoid memory buildup
        submit_batch_size = max_workers * 10  # Submit 10x workers at a time
        active_futures = {}
        task_refs = {}  # Keep minimal task info for error reporting

        with tqdm(total=total_combinations, desc="Calculating distances") as pbar:
            # Submit initial batch
            for _ in range(min(submit_batch_size, total_combinations)):
                try:
                    task = next(task_gen)
                    future = executor.submit(process_with_progress, task)
                    active_futures[future] = task[
                        0
                    ]  # Store apartment_id for error reporting
                    task_refs[future] = (task[0], task[2])  # (apt_id, school_id)
                except StopIteration:
                    break

            # Process completed tasks and submit new ones
            while active_futures:
                for future in as_completed(active_futures):
                    try:
                        result = future.result()
                        with results_lock:
                            results_buffer.append(result)
                            results_count += 1

                            # Flush to CSV when buffer reaches batch_size
                            if len(results_buffer) >= batch_size:
                                flush_results_to_csv()

                        pbar.update(1)

                        # Log progress periodically
                        if results_count % batch_size == 0:
                            logger.info(
                                f"Progress: {results_count:,}/{total_combinations:,} "
                                f"({100 * results_count / total_combinations:.1f}%)"
                            )
                    except Exception as e:
                        apt_id, school_id = task_refs[future]
                        logger.warning(
                            f"Error processing apartment {apt_id} to school {school_id}: {e}"
                        )
                        pbar.update(1)
                    finally:
                        # Remove completed future
                        del active_futures[future]
                        del task_refs[future]

                    # Submit new task if available
                    try:
                        task = next(task_gen)
                        new_future = executor.submit(process_with_progress, task)
                        active_futures[new_future] = task[0]
                        task_refs[new_future] = (task[0], task[2])
                    except StopIteration:
                        pass

        # Flush any remaining results
        with results_lock:
            flush_results_to_csv()

    logger.info(f"Completed distance calculations. Total results: {results_count:,}")

    # Reset rate limiter
    _rate_limiter = None

    # Clean up references
    del valid_apartments, school_ids, school_names, school_lats, school_lons
    del apt_ids, apt_lats, apt_lons
    gc.collect()

    # Read back with optimized dtypes for compatibility
    # Use chunked reading for very large files, but for most cases read all at once with optimized types
    logger.info(f"Loading results from {output_path} with optimized dtypes")
    try:
        distances_df = pd.read_csv(
            output_path,
            dtype={
                "distance_m": np.float32,
                "duration_s": np.float32,
                "school_id": np.int32,
            },
        )
        return distances_df
    except Exception as e:
        logger.warning(f"Could not load results DataFrame: {e}")
        return pd.DataFrame()


def find_closest_schools(
    distances_df: pd.DataFrame, output_path: str, top_n: int = 1
) -> pd.DataFrame:
    """
    Find the closest school(s) for each apartment.

    Args:
        distances_df: DataFrame with all apartment-school distances
        output_path: Path to save the closest schools CSV
        top_n: Number of closest schools to keep per apartment

    Returns:
        DataFrame with closest schools per apartment
    """
    logger.info(f"Finding closest {top_n} school(s) for each apartment")

    # Filter to successful calculations only (use view, not copy)
    valid_mask = distances_df["distance_m"].notna()
    valid_distances = distances_df[valid_mask]

    if len(valid_distances) == 0:
        logger.warning("No valid distances found!")
        return pd.DataFrame()

    # Sort by distance and get top N per apartment (more memory efficient)
    # Use nsmallest for better performance on large datasets
    closest = (
        valid_distances.sort_values("distance_m", kind="mergesort")
        .groupby("TRANSACTION_ID", sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Add rank column efficiently
    closest["rank"] = closest.groupby("TRANSACTION_ID", sort=False).cumcount() + 1

    # Optimize data types before saving
    if "distance_m" in closest.columns:
        closest["distance_m"] = closest["distance_m"].astype(np.float32)
    if "duration_s" in closest.columns:
        closest["duration_s"] = closest["duration_s"].astype(np.float32)
    if "school_id" in closest.columns:
        closest["school_id"] = closest["school_id"].astype(np.int32)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    logger.info(f"Saving closest schools to {output_path}")
    closest.to_csv(output_path, index=False)

    return closest


def main(
    limit_apartments: Optional[int] = None,
    limit_schools: Optional[int] = None,
    save_closest: bool = True,
    max_workers: int = 10,
    rate_limit: Optional[int] = None,
    use_geodesic: bool = False,
    grid_size_km: float = 10.0,
    max_distance_km: Optional[float] = None,
    seed: Optional[int] = None,
    size: Optional[int] = None,
):
    """
    Main function to calculate apartment-to-school distances.

    Args:
        limit_apartments: Limit number of apartments (for testing)
        limit_schools: Limit number of schools (for testing)
        save_closest: Whether to also save closest school per apartment
        max_workers: Number of threads to use for parallel processing
        rate_limit: Optional rate limit for concurrent API requests (defaults to max_workers)
        use_geodesic: If True, use geodesic (straight-line) distance instead of road distance
        grid_size_km: Size of grid cells in kilometers for spatial filtering (default: 10km)
        max_distance_km: Maximum distance to consider in kilometers (None = no limit beyond grid)
        seed: Random seed (matches training_tabpfn.py seed, default: 42)
        size: Dataset size to match (matches training_tabpfn.py size, default: 10000)
    """
    # Enable limit_apartments only for the same seed and size as training_tabpfn.py
    # training_tabpfn.py uses: size=10000, seed=42 (implicit default)
    TRAINING_TABPFN_SEED = 42
    TRAINING_TABPFN_SIZE = 10000
    
    # Set defaults to match training_tabpfn.py if not provided
    if seed is None:
        seed = TRAINING_TABPFN_SEED
    if size is None:
        size = TRAINING_TABPFN_SIZE
    
    # Enable limit_apartments only when seed and size match training_tabpfn.py
    if seed == TRAINING_TABPFN_SEED and size == TRAINING_TABPFN_SIZE:
        if limit_apartments is None:
            limit_apartments = TRAINING_TABPFN_SIZE
            logger.info(
                f"Auto-enabled limit_apartments={TRAINING_TABPFN_SIZE} "
                f"to match training_tabpfn.py (seed={seed}, size={size})"
            )
    # Paths
    school_csv_path = "datasets/filename=InstReg-udtraek17-01-2026.csv"
    apartment_csv_path = "datasets/Resights_Hackathon_Ejerlejligheder_TRAIN.csv"
    output_path = "datasets/apartment_school_distances.csv"
    closest_output_path = "datasets/apartment_closest_school.csv"

    # Check if files exist
    if not os.path.exists(school_csv_path):
        raise FileNotFoundError(f"School CSV not found: {school_csv_path}")
    if not os.path.exists(apartment_csv_path):
        raise FileNotFoundError(f"Apartment CSV not found: {apartment_csv_path}")

    # Parse schools data
    schools_df = parse_schools_data(school_csv_path)
    if limit_schools:
        schools_df = schools_df.head(limit_schools)
        logger.info(f"Limited to {limit_schools} schools for testing")

    # Load apartments data with optimized dtypes
    logger.info(f"Loading apartments data from {apartment_csv_path}")
    # Read CSV and optimize dtypes
    apartments_df = pd.read_csv(apartment_csv_path)

    # Convert coordinate columns to float32 to save memory
    if "LAT" in apartments_df.columns:
        apartments_df["LAT"] = pd.to_numeric(
            apartments_df["LAT"], errors="coerce", downcast="float"
        )
    if "LNG" in apartments_df.columns:
        apartments_df["LNG"] = pd.to_numeric(
            apartments_df["LNG"], errors="coerce", downcast="float"
        )

    # Filter to apartments with valid coordinates (use view, not copy)
    valid_mask = apartments_df[["LAT", "LNG"]].notna().all(axis=1)
    apartments_df = apartments_df[valid_mask]
    if limit_apartments:
        apartments_df = apartments_df.head(limit_apartments)
        logger.info(f"Limited to {limit_apartments} apartments for testing")

    logger.info(f"Loaded {len(apartments_df)} apartments with valid coordinates")

    # Note: Grid filtering will significantly reduce the number of pairs to process
    total_combinations_without_grid = len(apartments_df) * len(schools_df)
    logger.info(
        f"Total possible apartment-school pairs: {total_combinations_without_grid:,} "
        f"(will be filtered using {grid_size_km}km grid cells)"
    )

    # Adjust output path based on distance type
    if use_geodesic:
        base_name = output_path.replace(".csv", "_geodesic.csv")
        output_path = base_name
        closest_output_path = closest_output_path.replace(".csv", "_geodesic.csv")

    # Calculate distances with grid-based filtering
    calculate_apartment_school_distances(
        apartments_df,
        schools_df,
        output_path,
        max_workers=max_workers,
        rate_limit=rate_limit,
        use_geodesic=use_geodesic,
        grid_size_km=grid_size_km,
        max_distance_km=max_distance_km,
    )

    # Clean up DataFrames to free memory
    del apartments_df, schools_df
    gc.collect()

    # Print summary statistics using chunked reading to avoid loading entire file
    logger.info("\n=== Summary Statistics ===")
    total_pairs = 0
    successful = 0
    failed = 0
    distance_sum = 0.0
    duration_sum = 0.0
    valid_distances = 0
    valid_durations = 0

    # Read CSV in chunks for memory efficiency
    chunk_size = 10000
    for chunk in pd.read_csv(
        output_path,
        chunksize=chunk_size,
        dtype={"distance_m": "float32", "duration_s": "float32", "school_id": "int32"},
    ):
        total_pairs += len(chunk)
        successful += chunk["distance_m"].notna().sum()
        failed += chunk["distance_m"].isna().sum()
        valid_chunk = chunk[chunk["distance_m"].notna()]
        if len(valid_chunk) > 0:
            distance_sum += valid_chunk["distance_m"].sum()
            valid_distances += len(valid_chunk)
            if "duration_s" in valid_chunk.columns:
                valid_durations_chunk = valid_chunk[valid_chunk["duration_s"].notna()]
                if len(valid_durations_chunk) > 0:
                    duration_sum += valid_durations_chunk["duration_s"].sum()
                    valid_durations += len(valid_durations_chunk)

    logger.info(f"Total apartment-school pairs: {total_pairs:,}")
    logger.info(f"Successful distance calculations: {successful:,}")
    logger.info(f"Failed calculations: {failed:,}")

    if valid_distances > 0:
        mean_distance = distance_sum / valid_distances
        logger.info(f"Mean distance: {mean_distance:.0f} meters")
        if valid_durations > 0:
            mean_duration = duration_sum / valid_durations
            logger.info(
                f"Mean duration: {mean_duration:.0f} seconds "
                f"({mean_duration / 60:.1f} minutes)"
            )

    logger.info(f"\nResults saved to: {output_path}")

    # Find closest schools (will read CSV again, but that's necessary for the operation)
    if save_closest:
        # Read distances for closest school calculation
        distances_df = pd.read_csv(
            output_path,
            dtype={
                "distance_m": "float32",
                "duration_s": "float32",
                "school_id": "int32",
            },
        )
        closest_df = find_closest_schools(distances_df, closest_output_path)
        del distances_df
        gc.collect()
        if len(closest_df) > 0:
            logger.info(f"\nClosest schools saved to: {closest_output_path}")
            rank1 = closest_df[closest_df["rank"] == 1]
            if len(rank1) > 0:
                logger.info(
                    f"Mean distance to closest school: "
                    f"{rank1['distance_m'].mean():.0f} meters"
                )
        del closest_df
        gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate distances from apartments to schools"
    )
    parser.add_argument(
        "--geodesic",
        action="store_true",
        help="Use geodesic (straight-line) distance instead of road distance. "
        "Much faster but less accurate for actual travel.",
    )
    parser.add_argument(
        "--limit-apartments",
        type=int,
        default=None,
        help="Limit number of apartments (for testing)",
    )
    parser.add_argument(
        "--limit-schools",
        type=int,
        default=None,
        help="Limit number of schools (for testing)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Number of threads for parallel processing (default: 10)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Max concurrent API requests (only for road distance)",
    )
    parser.add_argument(
        "--no-closest",
        action="store_true",
        help="Don't save closest school per apartment",
    )
    parser.add_argument(
        "--grid-size-km",
        type=float,
        default=25.0,
        help="Size of grid cells in kilometers for spatial filtering (default: 25.0)",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=None,
        help="Maximum distance to consider in kilometers (None = no limit beyond grid filtering)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (matches training_tabpfn.py seed, default: 42)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Dataset size to match (matches training_tabpfn.py size, default: 10000)",
    )

    args = parser.parse_args()

    main(
        limit_apartments=args.limit_apartments,
        limit_schools=args.limit_schools,
        save_closest=not args.no_closest,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit,
        use_geodesic=args.geodesic,
        grid_size_km=args.grid_size_km,
        max_distance_km=args.max_distance_km,
        seed=args.seed,
        size=args.size,
    )
