import numpy as np

def assess_risk(flood_mask):
    flood_percentage = np.mean(flood_mask)
    
    if flood_percentage < 0.1:
        return "Low", flood_percentage
    elif flood_percentage < 0.3:
        return "Medium", flood_percentage
    else:
        return "High", flood_percentage
