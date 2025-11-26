#!/bin/bash

# Script to move Kaggle cache from home directory to /tmp2 and create symbolic link
# This helps avoid filling up home directory quota

echo "=========================================="
echo "Setting up Kaggle cache in /tmp2"
echo "=========================================="

# Define paths
HOME_CACHE="$HOME/.cache/kagglehub"
TMP2_CACHE="/tmp2/$USER/.cache"
TMP2_KAGGLE="$TMP2_CACHE/kagglehub"

# Step 1: Create cache directory in /tmp2
echo ""
echo "Step 1: Creating cache directory in /tmp2..."
mkdir -p "$TMP2_CACHE"
echo "Created: $TMP2_CACHE"

# Step 2: Remove old cache from home directory
if [ -d "$HOME_CACHE" ]; then
    echo ""
    echo "Step 2: Removing old Kaggle cache from home directory..."
    echo "Removing: $HOME_CACHE"
    rm -rf "$HOME_CACHE"
    echo "Old cache removed"
else
    echo ""
    echo "Step 2: No existing cache found in home directory"
fi

# Step 3: Create parent directory for symbolic link
echo ""
echo "Step 3: Preparing for symbolic link..."
mkdir -p "$HOME/.cache"

# Step 4: Create symbolic link
echo ""
echo "Step 4: Creating symbolic link..."
ln -sf "$TMP2_KAGGLE" "$HOME_CACHE"
echo "Symbolic link created: $HOME_CACHE -> $TMP2_KAGGLE"

# Verify
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Verification:"
echo "  Home cache link: $HOME_CACHE"
echo "  Points to: $(readlink -f $HOME_CACHE 2>/dev/null || echo 'Link created')"
echo "  Tmp2 cache: $TMP2_CACHE"
echo ""
echo "Future Kaggle downloads will be stored in /tmp2"
echo "This saves space in your home directory quota"
echo ""

# Show disk usage
echo "Current disk usage:"
echo "  Home directory: $(du -sh $HOME 2>/dev/null | cut -f1)"
echo "  /tmp2 directory: $(du -sh /tmp2/$USER 2>/dev/null | cut -f1)"
echo ""
