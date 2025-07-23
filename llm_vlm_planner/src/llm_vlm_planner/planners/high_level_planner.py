import re
import ast
import os
import yaml
import pathlib

import ollama
import chromadb


class HighLevelPlanner:
    def __init__(self, llm_client, config_path: str):
        """
        Initialize the high-level planner.
        
        Args:
            llm_client: The LLM client to use for planning
            config_path: Path to the configuration file containing the prompt
        """
        self.llm_client = llm_client
        self.config_path = config_path
        self.prompt_template = self._load_prompt_template()
        
        log_dir = pathlib.Path(__file__).parents[2] / "logs"
        self.log_file = log_dir / "high_level_planner.yaml"
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the configuration file."""
        with open(self.config_path, 'r') as f:
            return f.read()

    
    def plan(self, task: str, useful_docs: list):
        """
        Generate a high-level plan for the given task.
        
        Args:
            task: The semantic task to plan for (e.g., "grab the mug and put it in the box")
            
        Returns:
            A list of semantic primitives (e.g., ["grab the mug", "lift the mug", ...])
        """
        prompt = re.sub(r"TASK_PLACEHOLDER", task, self.prompt_template)

        #Ajout des document utile dans le prompt
        reasonning_response = self.llm_client.generate(prompt)
        response = re.sub(r'<think>.*?</think>\s*', '', reasonning_response, flags=re.DOTALL)
        
        # Create a dictionary to store the data
        log_data = {
            "task": task,
            "prompt": prompt,
            "reasoning_response": reasonning_response,
            "response": response
        }
        print(f"High-level plan response: {response}")
        
        # Write the data to a YAML file
        with open(self.log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        return ast.literal_eval(response)