from enum import Enum

class WorkflowType(Enum):
    LFQ_BENCH = "LFQ Bench"
    
    @property
    def description(self) -> str:
        return "Label-Free Quantification benchmark comparing two conditions (A vs B)"
    
    @property
    def requires_conditions(self) -> bool:
        return True