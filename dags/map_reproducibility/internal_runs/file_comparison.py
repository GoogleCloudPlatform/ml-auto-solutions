#!/usr/bin/env python3
import os
import filecmp
import difflib
from pathlib import Path


def compare_folders(folder1, folder2):
  """
  Compare two folders and report differences in files.

  Args:
      folder1 (str): Path to the first folder
      folder2 (str): Path to the second folder
  """
  folder1_path = Path(folder1)
  folder2_path = Path(folder2)

  # Check if folders exist
  if not folder1_path.exists() or not folder1_path.is_dir():
    print(f"Error: {folder1} is not a valid directory")
    return

  if not folder2_path.exists() or not folder2_path.is_dir():
    print(f"Error: {folder2} is not a valid directory")
    return

  # Get lists of files in both folders
  folder1_files = {f.name: f for f in folder1_path.glob("**/*") if f.is_file()}
  folder2_files = {f.name: f for f in folder2_path.glob("**/*") if f.is_file()}

  # Files only in folder1
  only_in_folder1 = set(folder1_files.keys()) - set(folder2_files.keys())
  if only_in_folder1:
    print(f"\nFiles only in {folder1}:")
    for filename in sorted(only_in_folder1):
      print(f"  - {filename}")

  # Files only in folder2
  only_in_folder2 = set(folder2_files.keys()) - set(folder1_files.keys())
  if only_in_folder2:
    print(f"\nFiles only in {folder2}:")
    for filename in sorted(only_in_folder2):
      print(f"  - {filename}")

  # Common files - check for differences
  common_files = set(folder1_files.keys()) & set(folder2_files.keys())
  different_files = []

  print("\nComparing common files...")
  for filename in sorted(common_files):
    file1_path = folder1_files[filename]
    file2_path = folder2_files[filename]

    # First quick check with filecmp
    if not filecmp.cmp(file1_path, file2_path, shallow=False):
      different_files.append(filename)

  # Print summary
  if not different_files:
    print("All common files are identical")
  else:
    print(f"\nFiles with differences ({len(different_files)}):")
    for filename in different_files:
      print(f"  - {filename}")
      show_diff(folder1_files[filename], folder2_files[filename])


def show_diff(file1_path, file2_path):
  """
  Show line differences between two files.

  Args:
      file1_path (Path): Path to the first file
      file2_path (Path): Path to the second file
  """
  try:
    with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
      file1_lines = file1.readlines()
      file2_lines = file2.readlines()

      diff = list(
          difflib.unified_diff(
              file1_lines,
              file2_lines,
              fromfile=str(file1_path),
              tofile=str(file2_path),
              lineterm="",
          )
      )

      if diff:
        print("\n    Differences:")
        # Only show first 10 diff lines to avoid overwhelming output
        for line in diff[:10]:
          print(f"      {line}")
        if len(diff) > 10:
          print(f"      ... and {len(diff)-10} more lines")
        print()
  except UnicodeDecodeError:
    print("    Cannot display diff (binary file or encoding issue)")
  except Exception as e:
    print(f"    Error displaying diff: {e}")


if __name__ == "__main__":
  # Pre-defined paths for comparison
  print(
      f"**********************Comparing folders recipes**********************"
  )
  folder1 = "../internal-gpu-recipes/recipes"
  folder2 = "./dags/map_reproducibility/recipes"

  print(f"Folder 1: {folder1}")
  print(f"Folder 2: {folder2}")

  compare_folders(folder1, folder2)

  print(f"**********************Comparing folders values**********************")
  folder1 = "../internal-gpu-recipes/values"
  folder2 = "./dags/map_reproducibility/values"
  print(f"Folder 1: {folder1}")
  print(f"Folder 2: {folder2}")

  compare_folders(folder1, folder2)

  print(
      f"**********************Comparing folders helm charts**********************"
  )
  folder1 = "../internal-gpu-recipes/src/helm-charts"
  folder2 = "./dags/map_reproducibility/helm-charts"
  print(f"Folder 1: {folder1}")
  print(f"Folder 2: {folder2}")

  compare_folders(folder1, folder2)
