def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def get_health_advisory(category):
    messages = {
        "Good": "Air quality is satisfactory. Minimal impact.",
        "Satisfactory": "Minor breathing discomfort to sensitive people.",
        "Moderate": "Breathing discomfort to people with lung disease.",
        "Poor": "Breathing discomfort to most people.",
        "Very Poor": "Respiratory illness on prolonged exposure.",
        "Severe": "Affects healthy people and serious health impact."
    }
    
    return messages.get(category, "")

    def predict_aqi(model, input_data):
        prediction = model.predict([input_data])[0]
        category = get_aqi_category(prediction)
        advisory = get_health_advisory(category)
    
        return prediction, category, advisory
    
   
    