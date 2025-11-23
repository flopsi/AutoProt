from enum import Enum
from typing import Optional

class Species(Enum):
    HUMAN = "human"
    ECOLI = "ecoli"
    YEAST = "yeast"
    
    @classmethod
    def from_protein_name(cls, protein_name: str) -> Optional['Species']:
        if not protein_name:
            return None
        protein_upper = str(protein_name).upper()
        if '_HUMAN' in protein_upper:
            return cls.HUMAN
        elif '_ECOLI' in protein_upper:
            return cls.ECOLI
        elif '_YEAST' in protein_upper:
            return cls.YEAST
        return None
    
    @property
    def color(self) -> str:
        from .colors import ThermoFisherColors
        return ThermoFisherColors.SPECIES_COLORS[self.value]
    
    @property
    def display_name(self) -> str:
        return self.value.capitalize()