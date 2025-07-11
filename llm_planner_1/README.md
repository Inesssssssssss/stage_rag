# LLM Planner for Robotics

A Python package for hierarchical task planning in robotics using Large Language Models (LLMs).

## Overview

This package implements a three-level planning approach:

1. **High-Level Planner**: Breaks down a semantic task (e.g., "grab the mug and put it in the box") into simpler semantic primitives (e.g., "grab the mug", "lift the mug", etc.).

2. **Outcome Planner**: Predicts the expected outcome for each semantic primitive (e.g., for "lift the mug", the outcome might be "having the mug in the gripper grabbed by the handle").

3. **Low-Level Planner**: Converts both the high-level plan and expected outcomes into executable Python code (e.g., `grab(mug, mug_subpart)`, `go_up()`, etc.).

A logical twin of the robot gripper validates the logical consistency of the low-level plan (e.g., can't grab an object if one is already in the gripper).

## Installation

```bash
# Install package in development mode
pip install -e .
```

## Dependencies

- Python 3.7+
- [Ollama](https://github.com/ollama/ollama) for accessing LLMs

## Usage Example

```python
from llm_planner.task_planner import TaskPlanner

# Initialize the task planner
planner = TaskPlanner(
    model_name="llama3",  # Specify the model you have in Ollama
)

# Generate a plan for a task
result = planner.plan("grab the mug and put it in the box")

# Print the results
print("High-level plan:")
for step in result["high_level_plan"]:
    print(f"- {step}")

print("\nExpected outcomes:")
for primitive, outcome in result["expected_outcomes"].items():
    print(f"- {primitive}: {outcome}")

print("\nLow-level plan:")
for step in result["low_level_plan"]:
    print(f"- {step}")
```

## Configuration

The planner uses prompt templates stored in configuration files:

- `config/high_level_prompt.txt`: Prompt for the high-level planner
- `config/outcome_prompt.txt`: Prompt for the outcome planner
- `config/low_level_prompt.txt`: Prompt for the low-level planner

You can customize these files to modify the system's behavior.