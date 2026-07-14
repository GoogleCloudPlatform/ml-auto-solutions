# check for required dependencies before running the DAG
check_deps = BashOperator(
    task_id='check_dependencies',
    bash_command='python3 -c "import transformers; import accelerate; import numpy; print(\'All dependencies present\')"'
)
