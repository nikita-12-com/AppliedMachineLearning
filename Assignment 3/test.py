import pytest
from score import score
import pickle
import os
import time 
import requests
from app import app 

model = pickle.load(open("best_model.pkl", "rb"))

def test_skore():
    try:
        sample_score = score("Sample text", model, 0.5)
    except Exception as exception:
        pytest.fail("Score function failed with error: {}".format(exception))
    
    assert len(sample_score) == 2, "Expected 2 output, found {}".format(len(sample_score))


def test_format():
    prediction, propensity = score("Sample text", model, 0.5)
    assert type(prediction) == bool, "Prediction type format check failed. Expected {}, Found {}".format(bool, type(prediction))
    assert type(propensity) == float, "Propensity type format check failed. Expected {}, Found {}".format(float, type(propensity))

def test_prediction_value():
    prediction, _ = score("Sample text", model, 0.5)
    assert prediction in (0, 1), "Prediction value is supposed to be either 0 or 1, Found {}".format(prediction)

def test_propensity_score():
    _, propensity = score("Sample text", model, 0.5) 
    assert propensity > 0 and propensity < 1, "Propensity value expected between 0 and 1, Found {}".format(propensity)

def test_prediction_with_zero_threshold():
    prediction, _ = score("Sample text", model, 0)
    assert prediction == 1, "Prediction expected 1 for Threshold 0, Prediction found: {}".format(prediction)


def test_prediction_with_one_threshold():
    prediction, _ = score("Sample text", model, 1)
    assert prediction == 0, "Prediction expected 0 for Threshold 1, Prediction found: {}".format(prediction)

def test_spam():
    spam_message = "Congratulations, you have been selected in the lucky draw and won 1 million dollars"
    prediction, _ = score(spam_message, model, 0.5)
    assert prediction == 1, "Prediction expected 1 for obvious spam message, Prediction found: {}".format(prediction)

def test_ham():
    ham_message = "I am scheduling a meeting with you tomorrow at 11 am as discussed in our call"
    prediction, _ = score(ham_message, model, 0.5)
    assert prediction == 0, "Prediction expected 0 for obvious ham message, Prediction found: {}".format(prediction)


"""Integration test for Flask API."""
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_score_endpoint(client):
    test_input = {
        "text": "Win a lottery now!",
        "threshold": 0.5
    }

    response = client.post("/score", json=test_input)
    
    assert response.status_code == 200
    
    data = response.get_json()
    assert "prediction" in data
    assert "propensity" in data
    assert isinstance(data["prediction"], bool) 
    assert isinstance(data["propensity"], float)