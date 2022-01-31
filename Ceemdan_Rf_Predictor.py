from PyEMD import CEEMDAN, Visualisation
import io
import talib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics

class CRPred:    
    
    def __init__(self, ts, horizon = 1, tail = 2000):
        self.prep = pd.DataFrame()
        self.feature_names = []
        self.imfs = []
        self.prep['time_series'] = ts.tail(tail)
        self.horizon = horizon
        
    def ceemdan(self, include = False, minus = 2):
        S1 = self.prep['time_series']
        S2 = S1.to_numpy()
        s = np.squeeze(np.asarray(S2))
        t = np.linspace(0,s.shape[0],s.shape[0])
        self.t = t
        self.not_included = minus
        
        ceemdan = CEEMDAN()
        cIMF = ceemdan.ceemdan(s,t)
        self.nIMF = cIMF.shape[0]
        
        imfr = np.zeros(cIMF.shape[1])
        for i in range(cIMF.shape[0] - self.not_included):
            imfr = imfr + cIMF[-(i+1)]
            
        self.prep['imf_rec'] = imfr
        self.feature_names = self.feature_names + ['imf_rec']
        
        if include:
            for n, cimf in enumerate(cIMF):
                self.prep['imf' + str(n + 1)] = cimf
                self.feature_names = self.feature_names + [
                    'imf' + str(n + 1)]
                self.imfs = self.imfs + [
                    'imf' + str(n + 1)]
                
    def _append(self, series, name):
            self.prep[name] = series
            
    def ma(self, periods, var = 'default'):
        if var == 'default':
            for n in periods:
                self.prep['ma' + str(n)] = talib.SMA(
                    self.prep['time_series'].values, timeperiod=n)
                self.feature_names = self.feature_names + [
                    'ma' + str(n)]
        else:
            for n in periods:
                self.prep['ma' + str(n) + '_' + var] = talib.SMA(
                    self.prep[var].values, timeperiod=n)
                self.feature_names = self.feature_names + [
                    'ma' + str(n) + '_' + var]
        
            
    def rsi(self, periods, var = 'default'):
        if var == 'default':
            for n in periods:
                self.prep['rsi' + str(n)] = talib.RSI(
                    self.prep['time_series'].values, timeperiod=n)
                self.feature_names = self.feature_names + [
                    'rsi' + str(n)]
        else:
            for n in periods:
                self.prep['rsi' + str(n) + '_' + var] = talib.RSI(
                    self.prep[var].values, timeperiod=n)
                self.feature_names = self.feature_names + [
                    'rsi' + str(n) + '_' + var]
            
    def freq(self, var, include = False):
        vis = Visualisation()
        tryal = np.asarray([self.prep[var].values,self.prep['time_series']])
        ifreq = vis._calc_inst_freq(tryal,self.t,False,None)[0]
        self.prep['freq_' + var] = ifreq
        if include:
            self.feature_names = self.feature_names + [
                    'freq_' + var]
        
    def train(self):

        self.prep['future'] = self.prep['time_series'].shift(-self.horizon)
        self.prep['pct_change'] = self.prep['future'].pct_change(self.horizon)
        self.prep.dropna(inplace=True)
        X = self.prep[self.feature_names]
        y = self.prep['pct_change']

        self.train_size = int(0.85 * y.shape[0])
        self.X_train = X[:self.train_size]
        self.y_train = y[:self.train_size]
        self.X_test = X[self.train_size:]
        self.y_test = y[self.train_size:]
        
        grid = {'n_estimators': [200],
                'max_depth': [2,3,7,10],
                'max_features': np.linspace(1, len(self.feature_names), 4, dtype = int),
                'random_state': [42]}
        test_scores = []
        rf_model = RandomForestRegressor()

        for g in ParameterGrid(grid):
            rf_model.set_params(**g) 
            rf_model.fit(self.X_train, self.y_train)
            test_scores.append(rf_model.score(self.X_test, self.y_test))

        best_index = np.argmax(test_scores)
        self.best_params = ParameterGrid(grid)[best_index]
        
    def fit(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            max_features=self.best_params['max_features'],
            random_state=self.best_params['random_state'])
        self.rf_model.fit(self.X_train, self.y_train)

        
        
    def test(self):
        y_pred = self.rf_model.predict(self.X_test)
        xi = self.prep['time_series']
        xi_test = xi[self.train_size:]
        price_pred = (y_pred + 1)*xi_test
        self.price_pred = price_pred
        fprice = self.prep['future']
        fprice_test = fprice[self.train_size:]
        self.fprice_test = fprice_test
        
        importances = self.rf_model.feature_importances_
        sorted_index = np.argsort(importances)[::-1]
        x_values = range(len(importances))
        labels = np.array(self.feature_names)[sorted_index]
        plt.bar(x_values, importances[sorted_index], tick_label=labels)
        plt.xticks(rotation=90)
        plt.title('Importances')
        plt.show()
        
        self.m_abs_err = metrics.mean_absolute_error(
            fprice_test.values, price_pred.values)
        self.mape = metrics.mean_absolute_percentage_error(
            fprice_test.values, price_pred.values)
        self.rm_sqr_err = np.sqrt(metrics.mean_squared_error(
            fprice_test.values, price_pred.values))
        
    def results(self):
        df = pd.DataFrame([[self.m_abs_err,
                            self.mape,
                            self.rm_sqr_err]],
                   columns=['Mean Absolute Error',
                            'Mean Absolute Percentage Error',
                            'Root Mean Squared Error'])
        return df
    
    def save_html(self, name):
        info = ('<h2>' + name  + '</h2>' + 
            '\n <p>Mean Absolute Error: ' + 
                str(self.m_abs_err) + '</p>' + 
            '\n <p>Mean Absolute Percentage Error: ' + 
                str(self.mape) + '</p>' + 
            '\n <p>Root Mean Squared Error: ' + 
                str(self.rm_sqr_err) + '</p> <br> <br> \n \n')
        arch = open('Resultados.txt', 'a')
        arch.write(info)
        arch.close()
        
    def save(self, name, history):
        df = pd.DataFrame([[self.m_abs_err,
                            self.mape,
                            self.rm_sqr_err]],
                   index=[name],
                   columns=['Mean Absolute Error',
                            'Mean Absolute Percentage Error',
                            'Root Mean Squared Error'])
        history = history.append(df)
        
        with pd.ExcelWriter("Historial de resultados.xlsx") as writer:
            history.to_excel(writer)
            
        return history
    
    def saveit(self, name, history, info, file_name):
        info.index = [name]
        history = history.append(info)
        with pd.ExcelWriter(file_name) as writer:
            history.to_excel(writer)
            
        return history
    
    def set_original(self):
        self.ma([14,30,50,200])
        self.rsi([14,30,50,200])
        self.train()
        self.fit()
        self.test()
        
        return self.results()
    
    def set0(self):
        self.ceemdan()
        self.freq('imf_rec')
        self.ma([14,30], 'freq_imf_rec')
        self.ma([14,30,50,200])
        self.rsi([14,30,50,200])
        self.ma([14,30,50,200], 'imf_rec')
        self.rsi([14,30,50,200], 'imf_rec')
        self.train()
        self.fit()
        self.test()
        
        return self.results()
        
    def set1_imfs(self):
        self.ceemdan(include = True)
        self.freq('imf_rec')
        for n, imf in enumerate(self.imfs):
            self.ma([50,200], imf)
        self.ma([7, 14, 30, 50, 200], 'freq_imf_rec')
        self.rsi([30,200], 'imf_rec')
        self.train()
        self.fit()
        self.test()
        
        return self.results()
        
    def best_today(self):
        self.ceemdan(include = True)
        for i in ['imf1', 'imf2', 'imf3']:
            self.feature_names.remove(i) 
        self.freq('imf_rec')
        for n, imf in enumerate(self.imfs[4:]):
            self.ma([14,30,50,200], imf)
        self.ma([7, 50], 'freq_imf_rec')
        self.rsi([30,200], 'imf_rec')
        self.ma([200])
        self.rsi([200])
        self.train()
        self.fit()
        self.test()
        
        return self.results()
    
    def set2_someimfs(self):
        self.ceemdan(include = True)
        for i in ['imf1', 'imf2', 'imf3']:
            self.feature_names.remove(i) 
        self.freq('imf_rec')
        for n, imf in enumerate(self.imfs[4:]):
            self.ma([14,30,50,200], imf)
        self.ma([7, 50], 'freq_imf_rec')
        self.rsi([30,200], 'imf_rec')
        self.train()
        self.fit()
        self.test()
        
        return self.results()