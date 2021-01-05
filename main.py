from classes.data import Data
from classes.forecast import Prediction, ProphetPrediction, NeuralProphetPrediction

def main():
    Covid_Data = Data()
    Covid_Data.preparing()
    Covid_Data.plot_data()
    Covid_Data.prep()
    # Predicting total cases Neural Prophet
    Neural = NeuralProphetPrediction(Covid_Data.get_Cases())
    # For next 7 days
    Neural.forecast(7)
    # Predicting total active cases Neural Prophet
    Neural = NeuralProphetPrediction(Covid_Data.get_Active())
    # For next 7 days
    Neural.forecast(7)
    # Predicting total cases Prophet
    Future = ProphetPrediction(Covid_Data.get_Cases())
    # For next 7 days
    Future.forecast(7)
    # Predicting total active cases Prophet
    Future = ProphetPrediction(Covid_Data.get_Active())
    # For next 7 days
    Future.forecast(7)

if __name__ == "__main__":
    main()
