"""
Calculate road distances from apartments to schools using OSRM API.

This module parses school data from InstReg CSV and calculates driving distances
from all apartments to all schools using the OSRM routing service.
"""

import os
import json
import time
import logging
import pandas as pd
import requests
from typing import Tuple, Optional
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

    # Convert coordinates from comma decimal separator to dot
    schools_df["lat"] = (
        schools_df["lat"].astype(str).str.replace(",", ".").astype(float)
    )
    schools_df["lon"] = (
        schools_df["lon"].astype(str).str.replace(",", ".").astype(float)
    )

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
) -> pd.DataFrame:
    """
    Calculate distances from all apartments to all schools using multithreading.

    Args:
        apartments_df: DataFrame with apartment data (must have LAT, LNG, TRANSACTION_ID)
        schools_df: DataFrame with school data (must have lat, lon, school_id)
        output_path: Path to save the results CSV
        max_workers: Maximum number of worker threads for parallel processing
        batch_size: Number of results to collect before saving progress (for large datasets)
        rate_limit: Optional rate limit (max concurrent requests). If None, uses max_workers.
        use_geodesic: If True, use geodesic (straight-line) distance instead of road distance.
                     Much faster but less accurate for actual travel.

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

    # Prepare all apartment-school pairs using vectorized operations
    logger.info("Preparing apartment-school pairs...")

    # Filter apartments with valid coordinates upfront
    valid_apartments = apartments_df.dropna(subset=["LAT", "LNG"]).copy()
    if len(valid_apartments) < len(apartments_df):
        skipped = len(apartments_df) - len(valid_apartments)
        logger.warning(f"Skipping {skipped} apartments with missing coordinates")

    # Pre-extract school data as numpy arrays for faster access
    school_ids = schools_df["school_id"].values
    school_names = schools_df["school_name"].values
    school_lats = schools_df["lat"].values
    school_lons = schools_df["lon"].values

    # Pre-extract apartment data - handle TRANSACTION_ID column if it exists
    if "TRANSACTION_ID" in valid_apartments.columns:
        apt_ids = valid_apartments["TRANSACTION_ID"].values
    else:
        apt_ids = valid_apartments.index.values
    apt_lats = valid_apartments["LAT"].values
    apt_lons = valid_apartments["LNG"].values

    # Create tasks with progress bar - iterate over apartments for better progress granularity
    tasks = []

    with tqdm(
        total=len(valid_apartments), desc="Preparing apartment-school pairs", unit="apt"
    ) as pbar:
        for apt_idx in range(len(valid_apartments)):
            apt_point = {"lat": apt_lats[apt_idx], "lon": apt_lons[apt_idx]}
            # Create all school pairs for this apartment at once
            apartment_tasks = [
                (
                    apt_ids[apt_idx],
                    apt_point,
                    school_ids[school_idx],
                    school_names[school_idx],
                    {"lat": school_lats[school_idx], "lon": school_lons[school_idx]},
                )
                for school_idx in range(len(schools_df))
            ]
            tasks.extend(apartment_tasks)
            pbar.update(1)

    total_combinations = len(tasks)
    logger.info(f"Total apartment-school pairs to process: {total_combinations:,}")

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
        """Flush buffered results to CSV file."""
        nonlocal file_initialized
        if not results_buffer:
            return

        # Convert buffer to DataFrame
        batch_df = pd.DataFrame(results_buffer)

        # Write to CSV (append mode after first write)
        if not file_initialized:
            batch_df.to_csv(output_path, index=False, mode="w")
            file_initialized = True
        else:
            batch_df.to_csv(output_path, index=False, mode="a", header=False)

        # Clear buffer
        results_buffer.clear()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_with_progress, task): task for task in tasks
        }

        # Process completed tasks with progress bar
        with tqdm(total=total_combinations, desc="Calculating distances") as pbar:
            for future in as_completed(future_to_task):
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
                    task = future_to_task[future]
                    logger.warning(
                        f"Error processing apartment {task[0]} to school {task[2]}: {e}"
                    )
                    pbar.update(1)

        # Flush any remaining results
        with results_lock:
            flush_results_to_csv()

    logger.info(f"Completed distance calculations. Total results: {results_count:,}")

    # Load the final CSV to return as DataFrame (optional, for compatibility)
    logger.info(f"Loading results from {output_path} for return")
    distances_df = pd.read_csv(output_path)

    # Reset rate limiter
    _rate_limiter = None

    return distances_df


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

    # Filter to successful calculations only
    valid_distances = distances_df[distances_df["distance_m"].notna()].copy()

    if len(valid_distances) == 0:
        logger.warning("No valid distances found!")
        return pd.DataFrame()

    # Sort by distance and get top N per apartment
    closest = (
        valid_distances.sort_values("distance_m")
        .groupby("TRANSACTION_ID")
        .head(top_n)
        .reset_index(drop=True)
    )

    # Add rank column
    closest["rank"] = closest.groupby("TRANSACTION_ID").cumcount() + 1

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
    """
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

    # Load apartments data
    logger.info(f"Loading apartments data from {apartment_csv_path}")
    apartments_df = pd.read_csv(apartment_csv_path)

    # Filter to apartments with valid coordinates
    apartments_df = apartments_df.dropna(subset=["LAT", "LNG"])
    if limit_apartments:
        apartments_df = apartments_df.head(limit_apartments)
        logger.info(f"Limited to {limit_apartments} apartments for testing")

    logger.info(f"Loaded {len(apartments_df)} apartments with valid coordinates")

    # Warn about computation size
    total_combinations = len(apartments_df) * len(schools_df)
    logger.info(
        f"Will calculate {total_combinations:,} apartment-school pairs. "
        f"This may take a while..."
    )

    # Adjust output path based on distance type
    if use_geodesic:
        base_name = output_path.replace(".csv", "_geodesic.csv")
        output_path = base_name
        closest_output_path = closest_output_path.replace(".csv", "_geodesic.csv")

    # Calculate distances
    distances_df = calculate_apartment_school_distances(
        apartments_df,
        schools_df,
        output_path,
        max_workers=max_workers,
        rate_limit=rate_limit,
        use_geodesic=use_geodesic,
    )

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total apartment-school pairs: {len(distances_df):,}")
    logger.info(
        f"Successful distance calculations: {distances_df['distance_m'].notna().sum():,}"
    )
    logger.info(f"Failed calculations: {distances_df['distance_m'].isna().sum():,}")

    if distances_df["distance_m"].notna().any():
        logger.info(f"Mean distance: {distances_df['distance_m'].mean():.0f} meters")
        logger.info(
            f"Mean duration: {distances_df['duration_s'].mean():.0f} seconds "
            f"({distances_df['duration_s'].mean() / 60:.1f} minutes)"
        )

    logger.info(f"\nResults saved to: {output_path}")

    # Find closest schools
    if save_closest:
        closest_df = find_closest_schools(distances_df, closest_output_path)
        if len(closest_df) > 0:
            logger.info(f"\nClosest schools saved to: {closest_output_path}")
            logger.info(
                f"Mean distance to closest school: "
                f"{closest_df[closest_df['rank'] == 1]['distance_m'].mean():.0f} meters"
            )


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
        default=10,
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

    args = parser.parse_args()

    main(
        limit_apartments=args.limit_apartments,
        limit_schools=args.limit_schools,
        save_closest=not args.no_closest,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit,
        use_geodesic=args.geodesic,
    )
