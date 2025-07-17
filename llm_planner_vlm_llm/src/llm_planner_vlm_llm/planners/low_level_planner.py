import re
import ast
import pathlib
import yaml
import os
from llm_planner_vlm_llm.utils.logical_twin import LogicalTwin

class LowLevelPlanner:
    def __init__(self, llm_client, config_path: str):
        """
        Initialize the low-level planner.
        
        Args:
            llm_client: The LLM client to use for planning
            config_path: Path to the configuration file containing the prompt
            logical_twin: A logical twin of the gripper for validation
        """
        self.llm_client = llm_client
        self.config_path = config_path
        self.logical_twin = LogicalTwin()
        self.prompt_template = self._load_prompt_template()
        
        log_dir = pathlib.Path(__file__).parents[2] / "logs"
        self.log_file = log_dir / "low_level_planner.yaml"
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the configuration file."""
        with open(self.config_path, 'r') as f:
            return f.read()

    
    def plan(self, task, plan, outcomes):
        """
        Generate a low-level plan for the given high-level plan and expected outcomes.
        
        Args:
            high_level_plan: The semantic primitives from the high-level planner
            outcomes: The expected outcomes for each primitive
            
        Returns:
            A list of low-level primitive calls as Python code strings
        """
        # Format the high-level plan and outcomes for the prompt
        
        log_data = {
            "task": task,
            "plan": plan,
            "outcomes": outcomes,
            "steps": [],
            "primitives": []
        }
        
        prompt = re.sub(r"TASK_PLACEHOLDER", str(task), self.prompt_template)
        prompt = re.sub(r"PLAN_PLACEHOLDER", str(plan), prompt)
        prompt = re.sub(r"OUTCOMES_PLACEHOLDER", str(outcomes), prompt)
        prompt = re.sub(r"PRIMITIVES_PLACEHOLDER", str(self.logical_twin.primitives), prompt)
        
        log_data["prompt"] = prompt
        
        self.logical_twin.reset()
        primitives = []
        #for plan_step in outcomes.keys():
        for plan_step in plan:
            self.llm_client.reset_chat()        
            success = False
            
            step_log = {
                "plan_step": plan_step,
                "attempts": []
            }
            
            chat_entry = re.sub(r"STEP_PLACEHOLDER", str(plan_step), prompt)
            while not success:
                reasonning_response = self.llm_client.chat(chat_entry)
                plan_step_primitives = re.sub(r'<think>.*?</think>\s*', '', reasonning_response, flags=re.DOTALL)
                
                attempt_log = {
                    "chat_entry": chat_entry,
                    "reasoning_response": reasonning_response,
                    "primitives": []
                }

                plan_step_primitives = ast.literal_eval(plan_step_primitives)
                    
                success_this_attempt = True
                for prim in plan_step_primitives:
                    try:
                        exec('self.logical_twin.' + prim)
                        attempt_log["primitives"].append({
                            "primitive": prim,
                            "success": True
                        })
                        primitives.extend(plan_step_primitives)
                        success = True
                    except Exception as e:
                        error_msg = str(e)
                        chat_entry = error_msg
                        attempt_log["primitives"].append({
                            "primitive": prim,
                            "success": False,
                            "error": error_msg
                        })
                        self.logical_twin.undo_action()
                        break
                    
                step_log["attempts"].append(attempt_log)
            
            log_data["steps"].append(step_log)
        
        log_data["primitives"] = primitives
        
        # Write the data to a YAML file
        with open(self.log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return primitives
