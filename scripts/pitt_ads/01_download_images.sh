#!/bin/bash

set -e

OUT_DIR="data/raw/pitt_ads/zips"
mkdir -p "$OUT_DIR"

for i in {0..10}
do
    echo "Downloading subfolder-$i.zip..."
    wget -c "https://storage.googleapis.com/ads-dataset/subfolder-$i.zip" \
        -O "$OUT_DIR/subfolder-$i.zip"
done

echo "Done downloading all image zip files."#!/bin/bash
#!/bin/bash

set -e

OUT_DIR="data/raw/pitt_ads/zips"
mkdir -p "$OUT_DIR"

for i in {0..10}
do
    echo "Downloading subfolder-$i.zip..."
    wget -c "https://storage.googleapis.com/ads-dataset/subfolder-$i.zip" \
        -O "$OUT_DIR/subfolder-$i.zip"
done

echo "Done downloading all image zip files."
set -e

OUT_DIR="data/raw/pitt_ads/zips"
mkdir -p "$OUT_DIR"

for i in {0..10}
do
    echo "Downloading subfolder-$i.zip..."
    wget -c "https://storage.googleapis.com/ads-dataset/subfolder-$i.zip" \
        -O "$OUT_DIR/subfolder-$i.zip"
done

echo "Done downloading all image zip files."
