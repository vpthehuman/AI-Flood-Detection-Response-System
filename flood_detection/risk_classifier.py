def assess_risk(flood_probability, precipitation_intensity):
    # Assuming flood_probability and precipitation_intensity are normalized between 0 and 1
    risk_score = 0.7 * flood_probability + 0.3 * precipitation_intensity
    
    if risk_score < 0.4:
        return "Low", risk_score
    elif risk_score < 0.7:
        return "Medium", risk_score
    else:
        return "High", risk_score
