class ThermoFisherColors:
    PRIMARY_RED = "#E71316"
    PRIMARY_GRAY = "#54585A"
    PRIMARY_WHITE = "#FFFFFF"
    LIGHT_GRAY = "#E2E3E4"
    NAVY = "#262262"
    DARK_RED = "#A6192E"
    ORANGE = "#EA7600"
    YELLOW = "#F1B434"
    GREEN = "#B5BD00"
    SKY = "#9BD3DD"
    PURPLE = "#8B4789"
    
    SPECIES_COLORS = {
        'human': NAVY,
        'ecoli': PURPLE,
        'yeast': ORANGE,
        'unknown': LIGHT_GRAY
    }
    
    CONDITION_COLORS = {
        'A': NAVY,
        'B': SKY,
    }
    
    CHART_PALETTE = [NAVY, ORANGE, GREEN, DARK_RED, YELLOW, SKY, PURPLE]
    BACKGROUND_LIGHT = "#F8F9FA"
    BACKGROUND_WHITE = PRIMARY_WHITE
    BORDER_LIGHT = LIGHT_GRAY
    BORDER_DARK = PRIMARY_GRAY
    
    @classmethod
    def get_species_color(cls, species: str) -> str:
        return cls.SPECIES_COLORS.get(species.lower(), cls.LIGHT_GRAY)
    
    @classmethod
    def get_condition_color(cls, condition: str) -> str:
        return cls.CONDITION_COLORS.get(condition.upper(), cls.PRIMARY_GRAY)