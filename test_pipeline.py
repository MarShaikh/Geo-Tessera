#!/usr/bin/env python3
"""
Quick test script to verify the pipeline works before running on full Nigeria.

Tests on a small region (Lagos State) with one year.
"""

import geopandas as gpd
from shapely.geometry import box
from geotessera import GeoTessera
from nigeria_pipeline import NigeriaEmbeddingPipeline


def create_test_boundary():
    """Create a simple test boundary for Lagos State."""
    # Lagos approximate bounds
    lagos_bounds = box(3.234180, 6.295058, 3.554180, 6.615057)


    gdf = gpd.GeoDataFrame(
        {
            'state_id': ['NG025'],
            'state_name': ['Lagos'],
            'geometry': [lagos_bounds]
        },
        crs="EPSG:4326"
    )

    return gdf


def test_api_connection():
    """Test basic GeoTessera API connection."""
    print("Testing GeoTessera API connection...")

    gt = GeoTessera()

    # Try fetching a single tile for Lagos
    try:
        # Lagos coordinates (approximate center)
        embedding, crs, transform = gt.fetch_embedding(lon=3.35, lat=6.55, year=2024)
        print(f"✓ Successfully fetched test tile")
        print(f"  Shape: {embedding.shape}")
        print(f"  CRS: {crs}")
        print(f"  Data type: {embedding.dtype}")
        return True
    except Exception as e:
        print(f"✗ Failed to fetch test tile: {e}")
        return False


def test_pipeline():
    """Test the full pipeline on a small region."""
    print("\n" + "="*60)
    print("Testing pipeline on Lagos State (2024 only)")
    print("="*60 + "\n")

    # Create test boundary
    test_boundary = create_test_boundary()
    print(f"Created test boundary for {test_boundary.iloc[0]['state_name']}")

    # Initialize pipeline
    pipeline = NigeriaEmbeddingPipeline(output_dir="./test_output")

    # Get bounds
    bounds = tuple(test_boundary.total_bounds)
    print(f"Bounds: {bounds}")

    # Check available years
    print("\nChecking available years...")
    years = pipeline.get_available_years(bounds)

    if not years:
        print("No data available for test region")
        return False

    print(f"Available years: {years}")

    # Test with just 2024 (or latest year)
    test_year = [max(years)]
    print(f"\nTesting with year: {test_year[0]}")

    # Extract features (optimized method)
    try:
        df = pipeline.extract_features_optimized(
            admin_boundaries=test_boundary,
            years=test_year,
            admin_level='state',
            id_col='state_id',
            name_col='state_name',
            statistics=['mean', 'std'],
            bounds=bounds
        )

        if df.empty:
            print("\n✗ Feature extraction returned empty dataframe!")
            print("This could mean:")
            print("  1. Polygon doesn't overlap with downloaded tiles")
            print("  2. CRS mismatch between polygon and rasters")
            print("  3. All pixels were masked/nodata")
            return False

        print("\n✓ Feature extraction successful!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.shape[1]}")
        print(f"\nFirst few values:")
        print(df.head())
        print(f"\nSample embedding values (first 5 bands):")
        print(df[['year', 'n_pixels', 'emb_mean_0', 'emb_mean_1', 'emb_mean_2', 'emb_mean_3', 'emb_mean_4']])

        # Save test results
        pipeline.save_results(df, "test_lagos_embeddings")

        print("\n" + "="*60)
        print("✓ Pipeline test PASSED!")
        print("="*60)
        print("\nYou can now run the full pipeline on all of Nigeria:")
        print("  python nigeria_pipeline.py --extract --lga-file ... --state-file ...")
        return True

    except Exception as e:
        print(f"\n✗ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("GeoTessera Nigeria Pipeline - Quick Test")
    print("="*60 + "\n")

    # Test 1: API connection
    api_ok = test_api_connection()

    if not api_ok:
        print("\n✗ API test failed. Check your internet connection and GeoTessera installation.")
        return

    # Test 2: Full pipeline
    pipeline_ok = test_pipeline()

    if pipeline_ok:
        print("\nTest output saved to: ./test_output/processed/")
        print("Check the CSV file to verify the structure looks correct.")


if __name__ == "__main__":
    main()
