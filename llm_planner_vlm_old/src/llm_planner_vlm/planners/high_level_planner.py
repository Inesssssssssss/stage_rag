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

    def get_useful_doc(self,collection,task):
        """
        Find the most useful information in the documents
        """
        response = ollama.embeddings(
            prompt=task,
            model="mxbai-embed-large"
            )
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=10
        )
        # Generate a threshold to filter relevant documents ( thresold can be adjusted)
        threshold = 0.9
        relevant_docs = []
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if dist <= threshold:
                relevant_docs.append(doc)
        return relevant_docs

    
    def plan(self, task: str, collection, image: str):
        """
        Generate a high-level plan for the given task.
        
        Args:
            task: The semantic task to plan for (e.g., "grab the mug and put it in the box")
            
        Returns:
            A list of semantic primitives (e.g., ["grab the mug", "lift the mug", ...])
        """
        docs = self.get_useful_doc(collection,task)
        print(f"useful docs : {docs}")
        prompt = re.sub(r"TASK_PLACEHOLDER", task, self.prompt_template)

        #Ajout des document utile dans le prompt
        prompt = re.sub(r"RAG", str(docs), prompt)
        reasonning_response = self.llm_client.generate_im(prompt,image)
        response = re.sub(r'<think>.*?</think>\s*', '', reasonning_response, flags=re.DOTALL)
        
        # Create a dictionary to store the data
        log_data = {
            "task": task,
            "prompt": prompt,
            "reasoning_response": reasonning_response,
            "response": response
        }
        
        # Write the data to a YAML file
        with open(self.log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        return ast.literal_eval(response)