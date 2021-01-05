import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

from neuralprophet import NeuralProphet

class Prediction(ABC):
    @abstractmethod
    def __init__(self, data):
        self.data = data
    @abstractmethod
    def forecast(self, days): 
        pass

class ProphetPrediction(Prediction):
    def __init__(self, data):
        self.data = data

    def forecast(self, days): 
        prophet = Prophet(changepoint_prior_scale=0.15, yearly_seasonality=True, interval_width=0.95)
        prophet.add_seasonality(name ='weekly', period = 7, fourier_order=10)
        prophet.fit(self.data)
        build_forecast = prophet.make_future_dataframe(periods= days, freq='D') 
        forecast = prophet.predict(build_forecast)

        data = forecast[['ds','yhat']].iloc[-(days):-1]
        data.set_index('ds', inplace=True, drop=True)

        ax = data.plot(grid=True, figsize=(15, 6), marker='o', title='Prediction')
        plt.show()
        print(data)

        change = data.diff()
        ax = change.plot(grid=True, figsize=(15, 6), marker='o', title='Change per Day Prediction')
        plt.show()
        print(change)

        prophet.plot(forecast, xlabel='Date', ylabel='Cases')
        plt.show()

        prophet.plot_components(forecast)
        plt.show()


class NeuralProphetPrediction(Prediction):
    def __init__(self, data):
        self.data = data

    def forecast(self, days): 
        prophet = NeuralProphet()
        prophet.fit(self.data, freq='D')
        build_forecast = prophet.make_future_dataframe(self.data, periods= days) 
        forecast = prophet.predict(build_forecast)

        data = forecast[['ds','yhat1']].iloc[-(days):-1]
        data.set_index('ds', inplace=True, drop=True)

        ax = data.plot(grid=True, figsize=(15, 6), marker='o', title='Neural Prediction')
        plt.show()
        print(data)

        change = data.diff()
        ax = change.plot(grid=True, figsize=(15, 6), marker='o', title='Change per Day Neural Prediction')
        plt.show()
        print(change)

        prophet.plot(forecast, xlabel='Date', ylabel='Cases')
        plt.show()

        prophet.plot_components(forecast)
        plt.show()
