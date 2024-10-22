import numpy as np

def assess_risk(flood_percentage):
    if flood_percentage < 0.1:
        return "Low"
    elif flood_percentage < 0.3:
        return "Medium"
    else:
        return "High"
