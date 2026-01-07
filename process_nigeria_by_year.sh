#!/bin/bash
#
# Process Nigeria embeddings year-by-year with incremental upload and cleanup
#
# Usage: ./process_nigeria_by_year.sh
#

set -e  # Exit on error

# Check environment variable
if [ -z "$AZURE_STORAGE_CONNECTION_STRING" ]; then
    echo "Error: AZURE_STORAGE_CONNECTION_STRING not set"
    echo "Run: export AZURE_STORAGE_CONNECTION_STRING='your-connection-string'"
    exit 1
fi

# Configuration
LGA_FILE="nigeria_boundaries/nga_admin2.shp"
STATE_FILE="nigeria_boundaries/nga_admin1.shp"
CONTAINER="nigeria-embeddings"
OUTPUT_DIR="nigeria_embeddings"

# Years to process (adjust as needed)
YEARS=(2017 2018 2019 2020 2021 2022 2023 2024)

echo "=========================================="
echo "Nigeria Embeddings Pipeline - By Year"
echo "=========================================="
echo "Years to process: ${YEARS[@]}"
echo "Container: $CONTAINER"
echo ""

# Process each year
for year in "${YEARS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Year: $year"
    echo "=========================================="

    # Step 1: Extract features for this year
    echo "Step 1: Extracting features for $year..."
    python nigeria_pipeline.py \
        --extract \
        --years "$year" \
        --lga-file "$LGA_FILE" \
        --state-file "$STATE_FILE" \
        --optimized \
        --keep-raw-tiffs \
        --output-dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo "Error: Extraction failed for year $year"
        exit 1
    fi

    echo "✓ Extraction complete for $year"

    # Step 2: Upload raw GeoTIFFs for this year
    echo ""
    echo "Step 2: Uploading raw GeoTIFFs for $year to Azure..."
    python nigeria_pipeline.py \
        --upload \
        --upload-raw \
        --azure-container "$CONTAINER" \
        --output-dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo "Error: Upload failed for year $year"
        exit 1
    fi

    echo "✓ Upload complete for $year"

    # Step 3: Cleanup temp files (keep processed CSVs)
    echo ""
    echo "Step 3: Cleaning up temp files for $year..."

    # Remove temp GeoTIFF directory for this year
    rm -rf "$OUTPUT_DIR/temp_tiffs_$year"
    echo "  Removed: $OUTPUT_DIR/temp_tiffs_$year"

    # Remove GeoTessera cache
    rm -rf global_0.1_degree_representation
    rm -rf global_0.1_degree_tiff_all
    echo "  Removed: GeoTessera cache"

    # Check disk space
    echo ""
    echo "Disk space after cleanup:"
    df -h . | grep -v Filesystem

    echo ""
    echo "✓ Year $year complete!"
    echo ""
done

echo ""
echo "=========================================="
echo "All Years Processed Successfully!"
echo "=========================================="
echo ""
echo "Processed CSVs are in: $OUTPUT_DIR/processed/"
echo "All raw GeoTIFFs uploaded to: $CONTAINER/raw_embeddings/"
echo ""
echo "Next steps:"
echo "1. Upload final CSVs (already uploaded incrementally)"
echo "2. Download CSVs from Azure for R analysis"
echo "3. Delete VM to stop charges"
echo ""
