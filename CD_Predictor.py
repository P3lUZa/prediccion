from PyEMD import CEEMDAN
import numpy as np
import pandas as pd

class CDPred:
    
    def __init__(self, ts, horizon = 1, min_data = 120, tail = 2000):
        self.data = ts.tail(tail)
        self.original = ts.tail(tail)
        self.horizon = horizon
        self.min_data = min_data
        self.interval = True
        self.standardize = False
        
    def ceemdan(self, imfs_omitted = 2):
        self.imfs_omitted = imfs_omitted
        S1 = self.data
        S2 = S1.to_numpy()
        s = np.squeeze(np.asarray(S2))
        t = np.linspace(0,s.shape[0],s.shape[0])
        self.t = t
        
        ceemdan = CEEMDAN()
        cIMF = ceemdan.ceemdan(s,t)
        self.nIMF = cIMF.shape[0]
        
        self.imfr = np.zeros(cIMF.shape[1])
        for i in range(cIMF.shape[0] - self.imfs_omitted):
            self.imfr = self.imfr + cIMF[-(i+1)]
            
        self.s_imfr = pd.Series(self.imfr)
        self.s_imfr.index = self.data.index
        self.data = self.s_imfr
        
    def _len_bin(self, length, index = -1):
        filter_1 = self.data <= min(
            (self.data.iloc[index] + length/2),
            self.data.max())
        filter_2 = self.data >= max(
            (self.data.iloc[index] - length/2),
            self.data.min())
        varbin = self.data.where(filter_1 & filter_2)
        varbin.dropna(inplace=True)
        return (varbin)
    
    def _ad_bin(self, index = -1, initial_length = 7, step = 7):
        lbin = self._len_bin(initial_length, index)
        i = 1
        while len(lbin) < self.min_data:
            lbin = self._len_bin(initial_length + i*step, index)
            i += 1
    
        return(lbin)
    
    def _bootstrap_mean(self, series, iterations = 1000):
        means = pd.Series(dtype = 'float64')
        for i in range(iterations):
            boot = series.sample(len(series), replace=True)
            means.at[i] = boot.mean()

        return (means.mean())
    
    def predict(self, indexes = [-1], interval = True):
        results = pd.DataFrame()
        for i, indx in enumerate(indexes):
            adbin = self._ad_bin(index = indx)
            if interval: 
                results.loc[i, 'index'] = indx
                results.loc[i, 'price'] = self.original.iloc[indx]
                diff = adbin.diff(-self.horizon)*(-1)
                results.loc[i, 'expected price'] = (self.data.iloc[indx] 
                                             + self._bootstrap_mean(diff))
                if interval:
                    results.loc[i, 'interval left limit'] = (self.data.iloc[indx] 
                                             + diff.quantile(0.025))
                    results.loc[i, 'interval right limit'] = (self.data.iloc[indx] 
                                             + diff.quantile(0.975))
                    
            else: 
                results.loc[i, 'index'] = indx
                results.loc[i, 'price'] = self.original.iloc[indx]
                results.loc[i, 'expected price'] = np.nan
                
        return results