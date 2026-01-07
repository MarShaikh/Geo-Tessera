#!/usr/bin/env python3
"""
Complete pipeline for downloading Nigeria embeddings (all years)
and extracting features by LGA and State level using GeoTessera Python API.

Usage:
    python nigeria_pipeline.py --extract --lga-file lgas.geojson --state-file states.geojson
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from geotessera import GeoTessera


class NigeriaEmbeddingPipeline:
    """Pipeline for downloading and processing Nigeria embeddings."""

    def __init__(self, output_dir: str = "./nigeria_embeddings"):
        self.gt = GeoTessera()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.processed_dir = self.output_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def get_nigeria_bounds(self, admin_boundaries: gpd.GeoDataFrame = None) -> tuple:
        """Get bounding box for Nigeria."""
        if admin_boundaries is not None:
            bounds = admin_boundaries.total_bounds
            return tuple(bounds)  # (minx, miny, maxx, maxy)
        else:
            # Nigeria approximate bounds
            return (2.6917, 4.2406, 14.6789, 13.8925)  # (min_lon, min_lat, max_lon, max_lat)

    def get_available_years(self, bounds: tuple) -> List[int]:
        """Check what years are available for the region."""
        print("Checking available years for Nigeria...")

        # Try common years
        test_years = list(range(2017, 2025))
        available_years = []

        for year in test_years:
            try:
                tiles = self.gt.registry.load_blocks_for_region(bounds=bounds, year=year)
                if tiles:
                    available_years.append(year)
                    print(f"  ✓ Year {year}: {len(tiles)} tiles")
            except Exception as e:
                print(f"  ✗ Year {year}: not available")

        return available_years

    def extract_features_for_polygons(
        self,
        admin_boundaries: gpd.GeoDataFrame,
        years: List[int],
        admin_level: str,
        id_col: str,
        name_col: str,
        statistics: List[str] = ['mean', 'std'],
        bounds: tuple = None
    ) -> pd.DataFrame:
        """
        Extract embedding statistics for each administrative polygon across all years.

        Parameters:
        -----------
        admin_boundaries : GeoDataFrame
            Administrative boundaries
        years : list
            Years to process
        admin_level : str
            'lga' or 'state'
        id_col : str
            Column name for admin ID
        name_col : str
            Column name for admin name
        statistics : list
            Statistics to compute: 'mean', 'std', 'median', 'min', 'max'
        bounds : tuple
            Bounding box for region (min_lon, min_lat, max_lon, max_lat)

        Returns:
        --------
        DataFrame with columns: admin_id, admin_name, year, emb_<stat>_<band>
        """
        print(f"\nExtracting {admin_level.upper()} features...")

        if bounds is None:
            bounds = self.get_nigeria_bounds(admin_boundaries)

        # Ensure admin boundaries are in WGS84
        admin_wgs84 = admin_boundaries.to_crs("EPSG:4326")

        all_results = []

        for year in years:
            print(f"\n--- Processing Year {year} ---")

            # Get all tiles for this year
            try:
                tiles_to_fetch = self.gt.registry.load_blocks_for_region(bounds=bounds, year=year)
                print(f"Found {len(tiles_to_fetch)} tiles for year {year}")
            except Exception as e:
                print(f"Error loading tiles for year {year}: {e}")
                continue

            if not tiles_to_fetch:
                print(f"No tiles found for year {year}")
                continue

            # Fetch all embeddings for this year
            print("Fetching embeddings...")
            tile_data = {}

            for year_val, tile_lon, tile_lat, embedding_array, crs, transform in tqdm(
                self.gt.fetch_embeddings(tiles_to_fetch),
                total=len(tiles_to_fetch),
                desc=f"Downloading tiles ({year})"
            ):
                # Store tile data with its geographic info
                tile_data[(tile_lon, tile_lat)] = {
                    'embedding': embedding_array,  # Shape: (height, width, 128)
                    'crs': crs,
                    'transform': transform
                }

            print(f"Downloaded {len(tile_data)} tiles")

            # Process each polygon
            print(f"Extracting features for {len(admin_wgs84)} {admin_level}s...")

            for idx, row in tqdm(admin_wgs84.iterrows(),
                                total=len(admin_wgs84),
                                desc=f"Processing {admin_level}s"):

                polygon_id = row[id_col]
                polygon_name = row[name_col]
                polygon_geom = row.geometry
                polygon_bounds = polygon_geom.bounds  # (minx, miny, maxx, maxy)

                # Find overlapping tiles
                overlapping_tiles = []
                for (tile_lon, tile_lat), data in tile_data.items():
                    # Each tile covers 0.1 degree
                    tile_minx, tile_miny = tile_lon, tile_lat
                    tile_maxx, tile_maxy = tile_lon + 0.1, tile_lat + 0.1

                    # Check intersection
                    if not (polygon_bounds[2] < tile_minx or polygon_bounds[0] > tile_maxx or
                           polygon_bounds[3] < tile_miny or polygon_bounds[1] > tile_maxy):
                        overlapping_tiles.append((tile_lon, tile_lat))

                if not overlapping_tiles:
                    continue

                # Extract pixels from all overlapping tiles
                all_pixels_per_band = [[] for _ in range(128)]

                for tile_lon, tile_lat in overlapping_tiles:
                    embedding = tile_data[(tile_lon, tile_lat)]['embedding']
                    transform_obj = tile_data[(tile_lon, tile_lat)]['transform']

                    # Get pixel coordinates within this tile that fall inside the polygon
                    height, width, n_bands = embedding.shape

                    for i in range(height):
                        for j in range(width):
                            # Convert pixel to geographic coordinates
                            lon, lat = transform_obj * (j, i)

                            # Check if point is in polygon
                            from shapely.geometry import Point
                            if polygon_geom.contains(Point(lon, lat)):
                                # Extract all band values for this pixel
                                pixel_values = embedding[i, j, :]
                                for band_idx in range(n_bands):
                                    all_pixels_per_band[band_idx].append(pixel_values[band_idx])

                # Compute statistics across all pixels
                if not all_pixels_per_band[0]:  # No pixels found
                    continue

                result = {
                    id_col: polygon_id,
                    name_col: polygon_name,
                    'year': year,
                    'n_pixels': len(all_pixels_per_band[0])
                }

                # Compute statistics for each band
                for band_idx in range(128):
                    pixels = np.array(all_pixels_per_band[band_idx])

                    if 'mean' in statistics:
                        result[f'emb_mean_{band_idx}'] = np.mean(pixels)
                    if 'std' in statistics:
                        result[f'emb_std_{band_idx}'] = np.std(pixels)
                    if 'median' in statistics:
                        result[f'emb_median_{band_idx}'] = np.median(pixels)
                    if 'min' in statistics:
                        result[f'emb_min_{band_idx}'] = np.min(pixels)
                    if 'max' in statistics:
                        result[f'emb_max_{band_idx}'] = np.max(pixels)

                all_results.append(result)

        df = pd.DataFrame(all_results)
        print(f"\n✓ Extracted features for {len(df)} {admin_level}-year combinations")
        return df

    def extract_features_optimized(
        self,
        admin_boundaries: gpd.GeoDataFrame,
        years: List[int],
        admin_level: str,
        id_col: str,
        name_col: str,
        statistics: List[str] = ['mean', 'std'],
        bounds: Optional[tuple] = None,
        keep_raw_tiffs: bool = False
    ) -> pd.DataFrame:
        """
        Optimized extraction using rasterio for masking - MUCH faster!
        Downloads tiles as GeoTIFFs first, then uses rasterio.mask.
        """
        print(f"\nExtracting {admin_level.upper()} features (optimized)...")

        if bounds is None:
            bounds = self.get_nigeria_bounds(admin_boundaries)

        admin_wgs84 = admin_boundaries.to_crs("EPSG:4326")

        all_results = []

        for year in years:
            print(f"\n--- Processing Year {year} ---")

            # Get tiles for this year
            try:
                tiles_to_fetch = self.gt.registry.load_blocks_for_region(bounds=bounds, year=year)
                print(f"Found {len(tiles_to_fetch)} tiles for year {year}")
            except Exception as e:
                print(f"Error loading tiles for year {year}: {e}")
                continue

            if not tiles_to_fetch:
                continue

            # Export to temporary GeoTIFFs
            temp_dir = self.output_dir / f"temp_tiffs_{year}"
            temp_dir.mkdir(exist_ok=True)

            print("Exporting tiles to GeoTIFF...")
            tiff_files = self.gt.export_embedding_geotiffs(
                tiles_to_fetch=tiles_to_fetch,
                output_dir=str(temp_dir),
                compress='lzw'
            )

            print(f"Exported {len(tiff_files)} GeoTIFF files")

            # Convert to Path objects for easier handling
            tiff_files = [Path(f) for f in tiff_files]

            # Verify files exist
            print(f"Verifying exported files...")
            existing_files = [f for f in tiff_files if f.exists()]
            print(f"  {len(existing_files)} / {len(tiff_files)} files exist")
            if len(existing_files) < len(tiff_files):
                print(f"  Warning: {len(tiff_files) - len(existing_files)} files missing!")

            tiff_files = existing_files

            # Now extract features using rasterio
            from rasterio.mask import mask
            import rasterio

            for idx, row in tqdm(admin_wgs84.iterrows(),
                               total=len(admin_wgs84),
                               desc=f"Extracting {admin_level} features"):

                polygon_id = row[id_col]
                polygon_name = row[name_col]
                polygon_geom_wgs84 = row.geometry

                # Collect pixels from all overlapping tiles
                all_band_pixels = [[] for _ in range(128)]
                tiles_processed = 0
                tiles_checked = 0

                for tiff_file in tiff_files:
                    try:
                        with rasterio.open(tiff_file) as src:
                            tiles_checked += 1

                            # Reproject polygon to raster CRS for intersection test
                            if src.crs and src.crs != admin_wgs84.crs:
                                from pyproj import Transformer
                                from shapely.ops import transform as shapely_transform

                                transformer = Transformer.from_crs(
                                    admin_wgs84.crs,
                                    src.crs,
                                    always_xy=True
                                )
                                polygon_geom_native = shapely_transform(
                                    transformer.transform,
                                    polygon_geom_wgs84
                                )
                            else:
                                polygon_geom_native = polygon_geom_wgs84

                            # Check if polygon intersects tile bounds (both in same CRS now)
                            tile_bounds_box = box(*src.bounds)
                            if not polygon_geom_native.intersects(tile_bounds_box):
                                continue

                            # Use the already-reprojected geometry
                            polygon_geom = [polygon_geom_native.__geo_interface__]

                            # Extract masked data
                            masked_data, out_transform = mask(
                                src,
                                polygon_geom,
                                crop=True,
                                all_touched=True,
                                nodata=src.nodata,
                                filled=False  # Get masked array
                            )

                            # masked_data shape: (128, height, width)
                            for band_idx in range(min(128, masked_data.shape[0])):
                                band_data = masked_data[band_idx]

                                # Handle masked array
                                if hasattr(band_data, 'mask'):
                                    valid = band_data.compressed()  # Get non-masked values
                                else:
                                    # Filter nodata
                                    if src.nodata is not None:
                                        valid = band_data[band_data != src.nodata]
                                    else:
                                        valid = band_data[~np.isnan(band_data)]

                                if len(valid) > 0:
                                    all_band_pixels[band_idx].extend(valid.flatten())

                            tiles_processed += 1

                    except Exception as e:
                        print(f"\n  Warning: Error processing {tiff_file.name}: {e}")
                        continue

                # Compute statistics
                if not all_band_pixels[0]:
                    print(f"\n  Warning: No pixels extracted for {polygon_name} (checked {tiles_checked} tiles, processed {tiles_processed} tiles)")
                    continue

                result = {
                    id_col: polygon_id,
                    name_col: polygon_name,
                    'year': year,
                    'n_pixels': len(all_band_pixels[0])
                }

                for band_idx in range(128):
                    pixels = np.array(all_band_pixels[band_idx])

                    if 'mean' in statistics:
                        result[f'emb_mean_{band_idx}'] = np.mean(pixels)
                    if 'std' in statistics:
                        result[f'emb_std_{band_idx}'] = np.std(pixels)
                    if 'median' in statistics:
                        result[f'emb_median_{band_idx}'] = np.median(pixels)
                    if 'min' in statistics:
                        result[f'emb_min_{band_idx}'] = np.min(pixels)
                    if 'max' in statistics:
                        result[f'emb_max_{band_idx}'] = np.max(pixels)

                all_results.append(result)

            # Clean up temp files or keep them
            if not keep_raw_tiffs:
                import shutil
                print(f"Cleaning up temporary GeoTIFF files...")
                shutil.rmtree(temp_dir)
            else:
                print(f"Raw GeoTIFF files saved in: {temp_dir}")

        df = pd.DataFrame(all_results)
        print(f"\n✓ Extracted features for {len(df)} {admin_level}-year combinations")
        return df

    def save_results(self, df: pd.DataFrame, filename: str):
        """Save results as CSV and optionally RDS."""
        # Save CSV
        csv_path = self.processed_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
        print(f"  Shape: {df.shape}")

        # Save RDS for R
        try:
            import pyreadr
            rds_path = self.processed_dir / f"{filename}.rds"
            pyreadr.write_rds(str(rds_path), df)
            print(f"✓ Saved: {rds_path}")
        except ImportError:
            print("  (Install pyreadr to save RDS format: pip install pyreadr)")

    def upload_to_azure_blob(
        self,
        container_name: str,
        connection_string: str = None,
        upload_raw_tiffs: bool = False,
        raw_tiffs_dir: Path = None
    ):
        """
        Upload processed files and optionally raw GeoTIFFs to Azure Blob Storage.

        Parameters:
        -----------
        container_name : str
            Azure blob container name
        connection_string : str, optional
            Connection string (uses env var if not provided)
        upload_raw_tiffs : bool
            If True, upload raw GeoTIFF files
        raw_tiffs_dir : Path, optional
            Directory containing raw GeoTIFF files to upload
        """
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            print("Azure SDK not installed. Run: pip install azure-storage-blob")
            return

        if connection_string is None:
            import os
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                print("Error: Set AZURE_STORAGE_CONNECTION_STRING environment variable")
                return

        print(f"\nUploading to Azure Blob Storage (container: {container_name})...")

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Create container if needed
        try:
            container_client.create_container()
            print("  Created container")
        except Exception:
            pass  # Container already exists

        # Upload processed CSV/RDS files
        print("\nUploading processed CSV/RDS files...")
        processed_count = 0
        for file_path in self.processed_dir.glob("*"):
            if file_path.is_file():
                blob_name = f"processed/{file_path.name}"
                with open(file_path, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data, overwrite=True)
                print(f"  ✓ {blob_name}")
                processed_count += 1

        print(f"✓ Uploaded {processed_count} processed files")

        # Upload raw GeoTIFF files if requested
        if upload_raw_tiffs and raw_tiffs_dir:
            print("\nUploading raw GeoTIFF files...")
            tiff_count = 0

            # Upload all TIFF files maintaining directory structure
            for tiff_file in raw_tiffs_dir.rglob("*.tif*"):
                # Preserve directory structure in blob storage
                relative_path = tiff_file.relative_to(raw_tiffs_dir)
                blob_name = f"raw_embeddings/{relative_path}"

                with open(tiff_file, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data, overwrite=True)

                tiff_count += 1
                if tiff_count % 10 == 0:
                    print(f"  Uploaded {tiff_count} files...")

            print(f"✓ Uploaded {tiff_count} raw GeoTIFF files")

        print("\n✓ Upload complete")

    def cleanup_local_files(self, keep_processed: bool = False):
        """
        Clean up local files to free disk space.

        Parameters:
        -----------
        keep_processed : bool
            If True, keep processed CSV/RDS files
        """
        import shutil

        print("\nCleaning up local files...")

        # Remove raw temp TIFF directories
        for temp_dir in self.output_dir.glob("temp_tiffs_*"):
            if temp_dir.is_dir():
                print(f"  Removing: {temp_dir}")
                shutil.rmtree(temp_dir)

        # Optionally remove processed files
        if not keep_processed:
            if self.processed_dir.exists():
                print(f"  Removing: {self.processed_dir}")
                shutil.rmtree(self.processed_dir)

        # Remove GeoTessera cache (raw .npy files)
        cache_dir = Path("global_0.1_degree_representation")
        if cache_dir.exists():
            print(f"  Removing GeoTessera cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        landmask_dir = Path("global_0.1_degree_tiff_all")
        if landmask_dir.exists():
            print(f"  Removing landmask cache: {landmask_dir}")
            shutil.rmtree(landmask_dir)

        print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Nigeria Embedding Pipeline")
    parser.add_argument("--extract", action="store_true", help="Extract features")
    parser.add_argument("--upload", action="store_true", help="Upload to Azure")
    parser.add_argument("--years", type=str, default="all",
                       help="Years: 'all' or comma-separated (e.g., '2020,2021,2022')")
    parser.add_argument("--lga-file", type=str, help="Path to LGA boundaries")
    parser.add_argument("--state-file", type=str, help="Path to State boundaries")
    parser.add_argument("--output-dir", type=str, default="./nigeria_embeddings")
    parser.add_argument("--azure-container", type=str, default="nigeria-embeddings")
    parser.add_argument("--statistics", type=str, default="mean,std",
                       help="Statistics: mean,std,median,min,max")
    parser.add_argument("--optimized", action="store_true",
                       help="Use optimized extraction (faster)")
    parser.add_argument("--keep-raw-tiffs", action="store_true",
                       help="Keep raw GeoTIFF files after extraction")
    parser.add_argument("--upload-raw", action="store_true",
                       help="Upload raw GeoTIFF files to Azure (requires --keep-raw-tiffs)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up all local files after upload")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = NigeriaEmbeddingPipeline(output_dir=args.output_dir)

    # Load boundaries
    if args.extract:
        if not args.lga_file and not args.state_file:
            print("Error: Provide --lga-file and/or --state-file")
            print("\nDownload Nigeria boundaries from:")
            print("  https://data.humdata.org/dataset/cod-ab-nga")
            return

        bounds = None

        # LGA extraction
        if args.lga_file:
            print(f"\nLoading LGA boundaries from {args.lga_file}...")
            lga_boundaries = gpd.read_file(args.lga_file)
            print(f"  Loaded {len(lga_boundaries)} LGAs")

            bounds = pipeline.get_nigeria_bounds(lga_boundaries)

            # Auto-detect column names
            lga_id_col = next((c for c in lga_boundaries.columns
                             if 'pcode' in c.lower() or 'id' in c.lower()),
                            lga_boundaries.columns[0])
            lga_name_col = next((c for c in lga_boundaries.columns
                               if 'name' in c.lower()),
                              lga_boundaries.columns[1])

            print(f"  Using ID column: {lga_id_col}")
            print(f"  Using name column: {lga_name_col}")

        # State extraction
        if args.state_file:
            print(f"\nLoading State boundaries from {args.state_file}...")
            state_boundaries = gpd.read_file(args.state_file)
            print(f"  Loaded {len(state_boundaries)} States")

            if bounds is None:
                bounds = pipeline.get_nigeria_bounds(state_boundaries)

            state_id_col = next((c for c in state_boundaries.columns
                               if 'pcode' in c.lower() or 'id' in c.lower()),
                              state_boundaries.columns[0])
            state_name_col = next((c for c in state_boundaries.columns
                                 if 'name' in c.lower()),
                                state_boundaries.columns[1])

            print(f"  Using ID column: {state_id_col}")
            print(f"  Using name column: {state_name_col}")

        # Get years
        if args.years == "all":
            years = pipeline.get_available_years(bounds)
        else:
            years = [int(y.strip()) for y in args.years.split(",")]

        print(f"\nProcessing years: {years}")

        statistics = [s.strip() for s in args.statistics.split(",")]
        print(f"Computing statistics: {statistics}")

        # Extract method
        extract_func = (pipeline.extract_features_optimized if args.optimized
                       else pipeline.extract_features_for_polygons)

        # Extract LGA features
        if args.lga_file:
            lga_df = extract_func(
                admin_boundaries=lga_boundaries,
                years=years,
                admin_level='lga',
                id_col=lga_id_col,
                name_col=lga_name_col,
                statistics=statistics,
                bounds=bounds,
                keep_raw_tiffs=args.keep_raw_tiffs
            )
            pipeline.save_results(lga_df, "nigeria_embeddings_lga_all_years")

        # Extract state features
        if args.state_file:
            state_df = extract_func(
                admin_boundaries=state_boundaries,
                years=years,
                admin_level='state',
                id_col=state_id_col,
                name_col=state_name_col,
                statistics=statistics,
                bounds=bounds,
                keep_raw_tiffs=args.keep_raw_tiffs
            )
            pipeline.save_results(state_df, "nigeria_embeddings_state_all_years")

    # Upload to Azure
    if args.upload:
        # Determine if we should upload raw TIFFs
        raw_tiffs_dir = None
        if args.upload_raw and args.keep_raw_tiffs:
            # Find the most recent temp_tiffs directory
            temp_dirs = list(pipeline.output_dir.glob("temp_tiffs_*"))
            if temp_dirs:
                raw_tiffs_dir = temp_dirs[0]  # Use first (or only) directory
                print(f"Will upload raw TIFFs from: {raw_tiffs_dir}")

        pipeline.upload_to_azure_blob(
            container_name=args.azure_container,
            upload_raw_tiffs=args.upload_raw,
            raw_tiffs_dir=raw_tiffs_dir
        )

    # Cleanup if requested
    if args.cleanup:
        # Keep processed files if not uploaded yet
        keep_processed = not args.upload
        pipeline.cleanup_local_files(keep_processed=keep_processed)

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
