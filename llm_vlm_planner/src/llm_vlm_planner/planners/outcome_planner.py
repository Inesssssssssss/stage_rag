import re
import ast
import os
import yaml
import pathlib

class OutcomePlanner:
    def __init__(self, llm_client, config_path: str):
        """
        Initialize the outcome planner.
        
        Args:
            llm_client: The LLM client to use for outcome prediction
            config_path: Path to the configuration file containing the prompt
        """
        self.llm_client = llm_client
        self.config_path = config_path
        self.prompt_template = self._load_prompt_template()
        
        log_dir = pathlib.Path(__file__).parents[2] / "logs"
        self.log_file = log_dir / "outcome_planner.yaml"
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the configuration file."""
        with open(self.config_path, 'r') as f:
            return f.read()
        
    
    def predict_outcomes(self, task, plan) -> dict:
        """
        Predict the expected outcome for each semantic primitive.
        
        Args:
            primitives: List of semantic primitives from the high-level planner
            
        Returns:
            A dictionary mapping each primitive to its expected outcome
        """

        prompt = re.sub(r"TASK_PLACEHOLDER", str(task), self.prompt_template)
        prompt = re.sub(r"PLAN_PLACEHOLDER", str(plan), prompt)
        
        reasonning_response = self.llm_client.generate(prompt)
            
        response = re.sub(r'<think>.*?</think>\s*', '', reasonning_response, flags=re.DOTALL)
        
        # Create a dictionary to store the data
        log_data = {
            "task": task,
            "plan": plan,
            "prompt": prompt,
            "reasoning_response": reasonning_response,
            "response": response
        }
        
        # Write the data to a YAML file
        with open(self.log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        return ast.literal_eval(response)