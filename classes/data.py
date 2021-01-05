import pandas as pd
import matplotlib. pyplot as plt
import numpy as np
from datetime import datetime


class Data:
    cdf = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    ddf = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    rdf = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    world = pd.DataFrame()
    def __init__(self):
        self.w_cases = pd.DataFrame()
        self.w_active = pd.DataFrame()
    @staticmethod
    def plotprep(df):
        numbers = df.iloc[:,4:-1]
        numbers2 = numbers.transpose()
        return numbers2
    @classmethod
    def preparing(cls):
        cdf1 = cls.cdf.drop(['Province/State','Country/Region','Lat', 'Long'], axis = 1)
        ddf1 = cls.ddf.drop(['Province/State','Country/Region','Lat', 'Long'], axis = 1)
        rdf1 = cls.rdf.drop(['Province/State','Country/Region','Lat', 'Long'], axis = 1)

        cdf1 = cls.cdf.agg(['sum'])
        ddf1 = cls.ddf.agg(['sum'])
        rdf1 = cls.rdf.agg(['sum'])

        cdf1 = cdf1.rename(index={'sum': 'cases'})
        ddf1 = ddf1.rename(index={'sum': 'deaths'})
        rdf1 = rdf1.rename(index={'sum': 'recoveries'})

        world = pd.concat([cdf1, ddf1,rdf1])
        cls.world = Data.plotprep(world)
        
        active = cls.world['cases'] - cls.world['deaths'] - cls.world['recoveries']
        cls.world['active cases'] = active
    @classmethod
    def plot_data(cls):
        plt.figure(figsize=(15, 6))
        plt.bar(cls.world.index, cls.world['active cases'], align='center', alpha=0.5, color = 'red')
        plt.ylabel('Estimate of active Cases')
        plt.title('Active cases')


        ax = cls.world.plot(grid=True, figsize=(15, 6), marker='o', title='World complete overview')
        plt.show()

    def prep(self):
        model = Data.world.copy()
        model2 = Data.world.copy()
        model['index'] = model.index
        model2['index'] = model2.index
        model.rename(columns={'index': 'ds', 'cases': 'y'}, inplace=True)
        model = model.drop(['deaths', 'recoveries', 'active cases'], axis=1)
        model2.rename(columns={'index': 'ds', 'active cases': 'y'}, inplace=True)
        model2 = model2.drop(['deaths', 'recoveries', 'cases'], axis=1)
        self.w_cases = model
        self.w_active = model2
        print(self.w_cases)
        print(self.w_active)
    def get_Cases(self):
        return self.w_cases
    def get_Active(self):
        return self.w_active
