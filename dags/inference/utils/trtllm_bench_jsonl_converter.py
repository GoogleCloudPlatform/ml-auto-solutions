import sys, glob, re, jsonlines


def read_input_file(file_path):
  with open(file_path, "r") as file:
    return file.read()


# Regex patterns to capture the data
patterns = {
    "engine_details": {
        "Model": r"Model:\s*(.+)",
        "Engine Directory": r"Engine Directory:\s*(.+)",
        "TensorRT-LLM Version": r"TensorRT-LLM Version:\s*(.+)",
        "Dtype": r"Dtype:\s*(.+)",
        "KV Cache Dtype": r"KV Cache Dtype:\s*(.+)",
        "Quantization": r"Quantization:\s*(.+)",
        "Max Input Length": r"Max Input Length:\s*(\d+)",
        "Max Sequence Length": r"Max Sequence Length:\s*(\d+)",
    },
    "runtime_info": {
        "TP Size": r"TP Size:\s*(\d+)",
        "PP Size": r"PP Size:\s*(\d+)",
        "Max Runtime Batch Size": r"Max Runtime Batch Size:\s*(\d+)",
        "Max Runtime Tokens": r"Max Runtime Tokens:\s*(\d+)",
        "Scheduling Policy": r"Scheduling Policy:\s*(.+)",
        "KV Memory Percentage": r"KV Memory Percentage:\s*([\d.]+)",
        "Issue Rate (req/sec)": r"Issue Rate \(req/sec\):\s*(.+)",
    },
    "statistics": {
        "Number of requests": r"Number of requests:\s*(\d+)",
        "Average Input Length (tokens)": r"Average Input Length \(tokens\):\s*(\d+)",
        "Average Output Length (tokens)": r"Average Output Length \(tokens\):\s*(\d+)",
        "Token Throughput (tokens/sec)": r"Token Throughput \(tokens/sec\):\s*([\d.e+-]+)",
        "Request Throughput (req/sec)": r"Request Throughput \(req/sec\):\s*([\d.e+-]+)",
        "Total Latency (ms)": r"Total Latency \(ms\):\s*([\d.e+-]+)",
    },
}


# Function to extract data based on regex patterns
def extract_data(patterns, data):
  extracted = {}
  for section, section_patterns in patterns.items():
    extracted[section] = {}
    for field, pattern in section_patterns.items():
      match = re.search(pattern, data)
      if match:
        extracted[section][field] = match.group(1)
  return extracted


def convert_to_jsonl(input_path, jsonl_path):
  input_data = read_input_file(input_path)
  extracted_data = extract_data(patterns, input_data)
  data = dict()
  data["dimensions"] = dict()
  data["metrics"] = dict()
  for sections in extracted_data.items():
    for key in sections[1]:
      try:
        float(sections[1][key])
        data["metrics"][key] = float(sections[1][key])
      except:
        data["dimensions"][key] = str(sections[1][key])
  if len(data["dimensions"]) == 0 or len(data["metrics"]) == 0:
    print(f"{input_path} contains incomplete results.")
  else:
    with jsonlines.open(jsonl_path, "a") as writter:
      writter.write(data)


file_pattern = "/scratch/*.txt"
file_paths = glob.glob(file_pattern)

if __name__ == "__main__":
  for file_path in file_paths:
    convert_to_jsonl(file_path, sys.argv[1])
