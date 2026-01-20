from sklearn.base import BaseEstimator as estimator

def score(text:str, model:estimator, threshold:float = 0.5) -> tuple[bool,float]:    
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")
    
    propensity = model.predict_proba([text])[0][1]

    prediction = propensity > threshold
    print(propensity, prediction)

    return bool(prediction.item()), float(propensity)