import json
import os
import pathlib

class LogicalTwinError(Exception):
    """Base exception class for LogicalTwin errors"""
    def __init__(self, message="An error occurred in the LogicalTwin"):
        self.message = message
        super().__init__(self.message)


class LogicalTwin:
    """
    A logical twin of a robot gripper to validate the logic of low-level plans.
    
    This class implements a simple state machine that tracks:
    - Whether the gripper is holding an object
    - The gripper's state (open/closed)
    - The gripper's position
    """
    
    def __init__(self):
        """Initialize the logical twin of the gripper."""
        self.reset()
        self.holding = None
        self.prev_holding = None
        
        package_dir = pathlib.Path(__file__).parent.absolute()
        
        with open(os.path.join(package_dir, 'primitives.json'), 'r') as f:
            self.primitives = json.load(f)

    def reset(self):
        """Reset the logical twin to its initial state."""
        self.holding = None
        
    def undo_action(self):
        self.holding = self.prev_holding
        
        
    def grasp_object(self, obj, part):
        primitive = self.primitives["grasp_object"]
        
        valid_objects = primitive["arguments"]["object"]
        if obj not in valid_objects:
            raise LogicalTwinError('Unknown object')
            
        valid_parts = primitive["arguments"]["subpart"][obj]
        if part not in valid_parts:
            raise LogicalTwinError(f'Unknown part for {obj}')
            
        if self.holding != None:
            raise LogicalTwinError('Gripper is already holding an object')
            
        self.prev_holding = self.holding
        self.holding = obj
        
    def drop_above(self, location):
        primitive = self.primitives["drop_above"]
            
        if self.holding == None:
            raise LogicalTwinError('Gripper is not holding an object')
            
        valid_locations = primitive["arguments"]["location"]
        if location not in valid_locations:
            raise LogicalTwinError('Unknown location')
            
        self.prev_holding = self.holding
        self.holding = None
        
    def handover(self, direction):
        primitive = self.primitives["handover"]
        
        if self.holding == None:
            raise LogicalTwinError('Gripper is not holding an object')
            
        valid_directions = primitive["arguments"]["direction"]
        if direction not in valid_directions:
            raise LogicalTwinError('Unknown direction')
            
        self.prev_holding = self.holding
        self.holding = None

