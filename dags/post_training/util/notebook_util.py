"""Utility functions for automating Jupyter notebooks in Airflow."""

from typing import Dict


def build_notebook_execution_command(
    notebook_path: str,
    parameters: Dict,
    maxtext_path: str = "$(pwd)",
    venv_path: str = None,
) -> str:
  """
  Build a shell command to execute a Jupyter notebook with parameter injection.

  This approach:
  1. Activates virtual environment (if provided)
  2. Modifies notebook with parameter injection
  3. Executes with jupyter nbconvert

  Args:
      notebook_path: Path to input notebook file
      parameters: Dictionary of parameters to inject (e.g., {"HF_TOKEN": "value"})
      maxtext_path: Path to MaxText repository
      venv_path: Path to virtual environment (relative to maxtext_path)

  Returns:
      Shell command string to execute notebook
  """
  # Build Python code to inject parameters (using our function logic)
  param_items = ", ".join([f'"{k}": {repr(v)}' for k, v in parameters.items()])

  # Activate venv if provided
  venv_activation = f"source {venv_path}/bin/activate\n" if venv_path else ""

  command = f"""
set -e

cd {maxtext_path}
{venv_activation}
# Inject parameters into notebook using Python
python << 'PYEOF'
import json
import re

# Parameters to inject
parameters = {{{param_items}}}

# Read original notebook
with open("{notebook_path}", "r") as f:
    nb = json.load(f)

modified = False

# Process each cell
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    source_lines = cell.get("source", [])
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    
    new_source = []
    for line in source_lines:
        new_line = line
        for param_key, param_value in parameters.items():
            pattern = rf'({{param_key}}\s*=\s*)(.+)$'
            match = re.search(pattern, line)
            if match:
                new_line = f'{{match.group(1)}}{{repr(param_value)}}\\n' if line.endswith('\\n') else f'{{match.group(1)}}{{repr(param_value)}}'
                modified = True
                print(f"✅ Replaced {{param_key}} in cell {{idx}}")
                break
        new_source.append(new_line)
    
    if new_source != source_lines:
        nb["cells"][idx]["source"] = new_source

if not modified:
    raise ValueError(f"Could not find parameter assignments in notebook")

# Write prepared notebook
with open("/tmp/notebook_with_params.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("✅ Prepared notebook with injected parameters")
PYEOF

# Execute the notebook
jupyter nbconvert --execute --to notebook \\
    --inplace /tmp/notebook_with_params.ipynb

echo "✅ Notebook execution completed successfully"
"""

  return command
