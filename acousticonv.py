# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:39:36 2021

@author: ddellong
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

# =============================================================================
### Sound Pressure and Levels in Pascals and dB formulas
# =============================================================================
class Acousticonv:
    
    def pascalize(dBval):
        """
        From dB to Pa
    
         Parameters
        ----------
        dBval : TYPE: float or int
            DESCRIPTION: Sound Pressure Level in dB
    
        Returns
        -------
        Pascval : TYPE float
            DESCRIPTION: Sound pressure in Pa
            underwater only as reference pressure is p0 = 1 µPa
        """
        
        Pascval = (10**(dBval/20))*(1*10**(-6))
        return Pascval
    
    def dBize(Pascals):
        """
         From Pa to dB
         Parameters
        ----------
        Pascals : TYPE: float or int
            DESCRIPTION: Sound Pressure in Pa
    
        Returns
        -------
        dBval : TYPE float
            DESCRIPTION: Sound Pressure Level in dB
            underwater only as reference pressure is p0 = 1 µPa
        """
        dBval = 20*np.log10(Pascals/10**(-6))
        return dBval
    
    def sumpress(p1,p2):
        """
        Parameters
        ----------
        p1 & P2 : TYPE: float or int
            DESCRIPTION: sound pressure in pascals
    
        Returns
        -------
        sumpre : TYPE float
            DESCRIPTION: sum of the pressures P1+P2 in dB
            underwater only as reference pressure is p0 = 1 µPa
        """
        sumpre = Acousticonv.dBize(np.sqrt((p1)**2 + (p2)**2))
        return sumpre
    
    def sumdBs(dB1,dB2):
        """
        Parameters
        ----------
        dB1 & dB2 : TYPE: float or int
            DESCRIPTION: sound pressure Level in dB
    
        Returns
        -------
        sumpre : TYPE float
            DESCRIPTION: sum of the Levels dB1+dB2 returned in dB
            Independent to the reference pressure
        """
        
        sumdBs = 10*np.log10((10**(dB1/10))+(10**(dB2/10)))
        return sumdBs
    
    
    
    def OneThirdOctave(fc, flist, vallist):
        """ 
        From unique frequencies To 1/3 Octave bands centered on fc
        Parameters
        ----------
        fc : TYPE: float or int
            Center frequency of the 1/3 octave (eg. 63 Hz)
        flist : TYPE: list or numpy array
            List of the original frequencies (e.g [50,63,80]) corresponding to the vallist
        vallist  : TYPE: list or numpy array
            List of the values in dB for each frequency of the flist  
        """
        f1 = fc/(2**(1/6))
        f2 = fc*(2**(1/6))
        nf = f2-f1
        
        fnew = np.arange(f1, f2+1, 1)
        
        RNlonlat2 = interp1d(flist, vallist, 'quadratic')
        val_champTiers63HzV2 = 10*np.log10(np.sum(10**(RNlonlat2(fnew)/10) * (fnew[1]-fnew[0])))
    
        return val_champTiers63HzV2
    
    
    def Octave(fc, flist, vallist):
        """ 
        From unique frequencies To Octave band centered on fc
        Parameters
        ----------
        fc : TYPE: float or int
            Center frequency of the octave (eg. 63 Hz)
        flist : TYPE: list or numpy array
            List of the original frequencies, corresponding to the vallist
        vallist  : TYPE: list or numpy array
            List of the values in dB for each frequency of the flist  
        """
        f1 = fc/(2**(1/2))
        f2 = fc*(2**(1/2))
        nf = f2-f1
        
        fnew = np.arange(f1, f2+1, 1)
        
        RNlonlat2 = interp1d(flist, vallist, 'quadratic')
        val_champTiers63HzV2 = 10*np.log10(np.sum(10**(RNlonlat2(fnew)/10) * (fnew[1]-fnew[0])))
    
        return val_champTiers63HzV2
    
    
class GridUtilities():
    def __init__(self):
        self.gg = ''
        
    def closest(lst, K): 
        '''
        Find the closest value to "K" in the list "lst"
            return the closest value of the list
        Parameters
        ----------
        lst : TYPE: list or 1D numpy array
            A list of values
        K : TYPE: float or int
            A single value
        '''
        
        lst1 = np.asarray(lst) 
        idx =  np.argmin(np.abs(lst1 - K))
        return lst1[idx]
    
    def closestnan(lst, K): 
        '''
        Find the closest value to "K" in the list "lst"
            return the closest value of the list.
            Allows to have NaN values in the list
        Parameters
        ----------
        lst : TYPE: list or 1D numpy array
            A list of values
        K : TYPE: float or int
            A single value
        '''
        lst1 = np.asarray(lst) 
        idx =  np.nanargmin(np.abs(lst1 - K))
        return lst1[idx]
    
    def closestime(lst, K):
        '''
        Find the closest value to "K" in the list "lst"
            return the closest value of the list.
            Allows to find the closest values in the datetime format
        Parameters
        ----------
        lst : TYPE: list or 1D numpy array
            A list of values
        K : TYPE: float or int
            A single value
        '''
        return min(lst, key=lambda x: abs(x - K))
    
    def CropArea(Lon_min, Lon_max, Lat_min, Lat_max, Lst_lons, Lst_lats, Matrix3, Dims=2):
        '''
        Extract a specific area from a geographic grid array.
         Parameters
        ----------
        Lst_lons : is longitude axis, list of longitudes
        Lst_lats : is the latitude axis, list of latitudes
        Matrix3  : is the Data matrix, can be up to a 4D array
        Dims     : dimensions of the Data array default Dims=2 
        
        return a 3 lists with: croped_lons 1d,  croped_lats 1d,  croped_datas 2d (Croped_datas 3d)
        '''
        if Lon_min < Lon_max and Lat_min < Lat_max :
            X_min = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_min))
            X_max = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_max))
            Y_min = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_min))
            Y_max = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_max))
        elif Lon_min > Lon_max  and Lat_min > Lat_max :
            X_min = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_max))
            X_max = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_min))
            Y_min = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_max))
            Y_max = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_min))
        elif Lon_min > Lon_max  and Lat_min < Lat_max :
            X_min = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_max))
            X_max = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_min))
            Y_min = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_min))
            Y_max = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_max))
        elif Lon_min < Lon_max  and Lat_min > Lat_max :
            X_min = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_min))
            X_max = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_max))
            Y_min = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_max))
            Y_max = Lst_lats.index(GridUtilities.closest(Lst_lats, Lat_min))        
        
        # Check that the function can't return an empty sequence of lon/lat or a one val array
        if Y_min == Y_max or Y_min == Y_max-1:
            Y_max = Y_max+1
            Y_min = Y_min-1
            
        if X_min == X_max  or X_min == X_max-1:
            X_max = X_max+1
            X_min = X_min-1
            
    
        if Dims == 4:
            return Lst_lons[X_min:X_max], Lst_lats[Y_min:Y_max], Matrix3[:, :, Y_min:Y_max, X_min:X_max]
        elif Dims == 3:
            return Lst_lons[X_min:X_max], Lst_lats[Y_min:Y_max], Matrix3[:, Y_min:Y_max, X_min:X_max]
        elif Dims == 2:
            return Lst_lons[X_min:X_max], Lst_lats[Y_min:Y_max], Matrix3[Y_min:Y_max, X_min:X_max]

    def CropList(Lon_min, Lon_max, Lst_lons):
        '''
        Extrat values from a list between 2 others
        Parameters
        ----------
        Lst_lons : TYPE: list or 1D numpy array
            A list of values
        Lon_min : TYPE: float or int
            A single value of the lower bondary
        Lon_max : TYPE: float or int
            A single value of the upper bondary
        '''
        X_min = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_min))
        X_max = Lst_lons.index(GridUtilities.closest(Lst_lons, Lon_max))
        return Lst_lons[X_min:X_max]
    
    def Get_Column(matrix, i):
        return [row[i] for row in matrix]
    
    def dms_to_dd(d, m, s, direction):
        dd = float(d) + float(m)/60 + float(s)/3600
        if direction == 'W' or direction == 'S': dd *= -1
        return dd
    
    def Convert_codes(IN_cds, CSV_CODE_FILE , COL_OUT=4, COL1=1, COL2=2, NAN=98, NAN_OUT=0):
        '''
        Function Specific to SHOM CABAT software
        Convert 2D or 1D array, which values are codes into 2D or 1D array with 
        values according to codes_table.csv file that as exactly 10 Rows 
        (1 for headers at the top and the last one for anything else in case netCDF file)
        - COL_OUT is the column where the new values are stored 
        - COL1 and COL2 are the codes values that need to be modified
        (COL1 and COL2 can be equal, if codes values are stored in the same collumn)
        COL_OUT=4, COL1=1, COL2=2 default values are set to transfrom codes array to Celerity array
        NAN values can be edited and NAN_OUT also...
        
        It return the same 2D or 1D array with modified values from the CSV file
        '''
        # Import CSV file using pandas
        #import pandas as pd
        
        df = pd.read_csv(CSV_CODE_FILE ,  sep=';', header=None)
        codes_table = df.values
        
        if IN_cds.ndim == 1:
            OUT = np.zeros((IN_cds.shape[0]))
            
            for i in range(IN_cds.shape[0]):
                if (IN_cds[i]>= float(codes_table[1][COL1])) and (IN_cds[i] <= float(codes_table[1][COL2])):
                    OUT[i] = float(codes_table[1][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[2][COL1])) and (IN_cds[i] <= float(codes_table[2][COL2])):
                    OUT[i] = float(codes_table[2][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[3][COL1])) and (IN_cds[i] <= float(codes_table[3][COL2])):
                    OUT[i] = float(codes_table[3][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[4][COL1])) and (IN_cds[i] <= float(codes_table[4][COL2])):
                    OUT[i] = float(codes_table[4][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[5][COL1])) and (IN_cds[i] <= float(codes_table[5][COL2])):
                    OUT[i] = float(codes_table[5][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[6][COL1])) and (IN_cds[i] <= float(codes_table[6][COL2])):
                    OUT[i] = float(codes_table[6][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[7][COL1])) and (IN_cds[i] <= float(codes_table[7][COL2])):
                    OUT[i] = float(codes_table[7][COL_OUT])
                elif (IN_cds[i]>= float(codes_table[8][COL1])) and (IN_cds[i] <= float(codes_table[8][COL2])):
                    OUT[i] = float(codes_table[8][COL_OUT])
                elif (IN_cds[i]>= NAN):
                    OUT[i] = NAN_OUT
                    
        elif IN_cds.ndim == 2:
            OUT = np.zeros((IN_cds.shape[0], IN_cds.shape[1]))
            
            for i in range(IN_cds.shape[0]):
                for j in range(IN_cds.shape[1]):
                    if (IN_cds[i][j]>= float(codes_table[1][COL1])) and (IN_cds[i][j] <= float(codes_table[1][COL2])):
                        OUT[i][j] = float(codes_table[1][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[2][COL1])) and (IN_cds[i][j] <= float(codes_table[2][COL2])):
                        OUT[i][j] = float(codes_table[2][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[3][COL1])) and (IN_cds[i][j] <= float(codes_table[3][COL2])):
                        OUT[i][j] = float(codes_table[3][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[4][COL1])) and (IN_cds[i][j] <= float(codes_table[4][COL2])):
                        OUT[i][j] = float(codes_table[4][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[5][COL1])) and (IN_cds[i][j] <= float(codes_table[5][COL2])):
                        OUT[i][j] = float(codes_table[5][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[6][COL1])) and (IN_cds[i][j] <= float(codes_table[6][COL2])):
                        OUT[i][j] = float(codes_table[6][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[7][COL1])) and (IN_cds[i][j] <= float(codes_table[7][COL2])):
                        OUT[i][j] = float(codes_table[7][COL_OUT])
                    elif (IN_cds[i][j]>= float(codes_table[8][COL1])) and (IN_cds[i][j] <= float(codes_table[8][COL2])):
                        OUT[i][j] = float(codes_table[8][COL_OUT])
                    elif (IN_cds[i][j]>= NAN):
                        OUT[i][j] = NAN_OUT
                    
        return OUT
    
#sumpress(0.002,0.002)
