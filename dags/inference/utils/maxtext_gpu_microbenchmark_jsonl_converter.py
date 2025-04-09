import re, sys, jsonlines


def extract_and_write_to_jsonl_pattern(input_file, output_file):
  """
  Extracts AutoRegressive results from a text file using patterns
  and writes them to a JSONL file.

  Args:
      input_file (str): Path to the input text file.
      output_file (str): Path to the output JSONL file.
  """
  extraction_patterns = {
      "ar_step_average_time_ms": r"AR step average time: (\d+\.\d+) ms",
      "ar_step_average_time_per_seq_ms": r"AR step average time per seq: (\d+\.\d+) ms",
      "ar_global_batch_size": r"AR global batch size: (\d+)",
      "ar_throughput_tokens_per_second": r"AR throughput: (\d+\.\d+) tokens/second",
      "ar_memory_bandwidth_gb_per_second": r"AR memory bandwidth per device: (\d+\.\d+) GB/s",
  }

  results = dict()
  results["dimensions"] = dict()
  results["metrics"] = dict()
  try:
    with open(input_file, "r") as f:
      for line in f:
        line = line.strip()
        for key, pattern in extraction_patterns.items():
          match = re.search(pattern, line)
          if match:
            if "." in match.group(1):
              results["metrics"][key] = float(match.group(1))
            else:
              results["metrics"][key] = int(match.group(1))
            break  # Move to the next line once a match is found

    if results:
      with jsonlines.open(output_file, "w") as writter:
        writter.write(results)
      print(f"Extracted results written to {output_file}")
    else:
      print("No AutoRegressive results found in the input file.")

  except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":
  input_file = "output.txt"  # Replace with the actual path
  try:
    output_file = sys.argv[1]
    extract_and_write_to_jsonl_pattern(input_file, output_file)
  except Exception as e:
    print(f"An error occurred: {e}")
