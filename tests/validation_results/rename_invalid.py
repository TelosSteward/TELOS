#!/usr/bin/env python3
"""Rename invalid validation reports."""
import os
import glob

# Keep only the latest report (233025) - mark all others as invalid
valid_timestamp = "20251107_233025"

# Get all report files
all_files = glob.glob("forensic_report_*")

for filename in all_files:
    # Skip the valid reports
    if valid_timestamp in filename:
        continue

    # Skip already marked invalid files
    if "(invalid Data)" in filename:
        # Fix incorrect extensions
        if filename.endswith(" (invalid Data).txt"):
            # Determine correct extension
            if ".json" in filename and not filename.startswith("forensic"):
                continue  # Skip weird files
            if "_debug.pkl" in filename:
                new_name = filename.replace(" (invalid Data).txt", " (invalid Data).pkl")
            elif ".json" in filename:
                new_name = filename.replace(".json (invalid Data).txt", " (invalid Data).json")
            else:
                new_name = filename  # Already correct

            if new_name != filename:
                print(f"Fixing: {filename} -> {new_name}")
                os.rename(filename, new_name)
        continue

    # Mark as invalid
    base, ext = os.path.splitext(filename)
    new_name = f"{base} (invalid Data){ext}"
    print(f"Marking invalid: {filename} -> {new_name}")
    os.rename(filename, new_name)

print("\n✅ All invalid reports renamed!")
