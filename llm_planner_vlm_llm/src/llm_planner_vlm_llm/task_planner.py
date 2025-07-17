import pathlib
import os
import ast
import ollama

from llm_planner_vlm_llm.planners.high_level_planner import HighLevelPlanner
from llm_planner_vlm_llm.planners.outcome_planner import OutcomePlanner
from llm_planner_vlm_llm.planners.low_level_planner import LowLevelPlanner
from llm_planner_vlm_llm.utils.llm_client import LLMClient


class TaskPlanner:
    """
    Main class that coordinates the three-level planning process:
    1. High-level planner breaks down a task into semantic primitives
    2. Outcome planner predicts the expected outcome of each primitive
    3. Low-level planner generates executable code with logical validation
    """
    
    def __init__(
        self, 
        model_name: str = "llama3.1"):
        """
        Initialize the task planner with its components.
        
        Args:
            model_name: The name of the LLM model to use
            high_level_config_path: Path to the high-level planner prompt
            outcome_config_path: Path to the outcome planner prompt
            low_level_config_path: Path to the low-level planner prompt
        """
        # Shared LLM client for all planners
        self.llm_client = LLMClient(model_name=model_name)
        
        package_dir = pathlib.Path(__file__).parent.absolute()
        
        log_dir = os.path.join(package_dir.parent, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        #high_level_config = os.path.join(package_dir, "config", "high_level_prompt.txt")
        high_level_config = os.path.join(package_dir, "config", "high_level_prompt_vlm.txt")
        outcome_config = os.path.join(package_dir, "config", "outcome_prompt.txt")
        low_level_config = os.path.join(package_dir, "config", "low_level_prompt.txt")
        
        # Initialize planners
        self.high_level_planner = HighLevelPlanner(
            llm_client=self.llm_client,
            config_path=high_level_config
        )
        
        self.outcome_planner = OutcomePlanner(
            llm_client=self.llm_client,
            config_path=outcome_config
        )
        
        self.low_level_planner = LowLevelPlanner(
            llm_client=self.llm_client,
            config_path=low_level_config
        )
    
    def plan(self, task: str, useful_docs: list):
        """
        Generate a complete plan for the given task.
        
        Args:
            task: The semantic task to plan for (e.g., "grab the mug and put it in the box")
            
        Returns:
            A dictionary containing:
            - high_level_plan: The list of semantic primitives
            - expected_outcomes: The expected outcomes for each primitive
            - low_level_plan: The executable Python code for each primitive
            
        """
        
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        print(f"\n{MAGENTA}{BOLD}=== 🧠 TASK PLANNER ==={RESET}")
        print(f"{MAGENTA}Task:{RESET} {task}")
     
        
        #Fancy print 
        high_level_plan = self.high_level_planner.plan(task, useful_docs)
        print(f"\n{CYAN}{BOLD}=== 📋 HIGH LEVEL PLAN ==={RESET}")
        for i, step in enumerate(high_level_plan):
            print(f"{GREEN}Step {i}:{RESET} {step}")
            

        expected_outcomes = self.outcome_planner.predict_outcomes(task, high_level_plan)
        print(f"\n{CYAN}{BOLD}=== 🎯 EXPECTED OUTCOMES ==={RESET}")
        for step, outcome in expected_outcomes.items():
            print(f"{YELLOW}• {step}:{RESET} {outcome}")

        low_level_plan = self.low_level_planner.plan(task, high_level_plan, expected_outcomes)
        print(f"\n{CYAN}{BOLD}=== 💻 LOW LEVEL PLAN ==={RESET}")
        for i, step in enumerate(low_level_plan):
            print(f"{BLUE}Step {i}:{RESET} {step}")
            
        print("\n")
        
        return low_level_plan