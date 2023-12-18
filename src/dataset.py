import glob

import numpy as np
import pandas as pd

import netCDF4 as nc
from ncBuilder import ncHelper
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline

def e(Td):
    return 6.112 * np.exp((17.67 * Td) / (Td + 243.5))


def load_base_v1(path):
    '''
    Hard-code to load some variables and calculate specific humidity at given
    temperature of dew point.
    
    Returns inputs, outputs and times, all prepared with channels, but not sample
    dimension
    
    Parameters
    ----------
    path : str
        String that contains the path to the .csv file.
        
    Returns
    -------
    inputs : np.ndarray
        Array with (n_steps, channels), being the n_steps the days of avaiable data and
        channels the number of variables used as input.
    outputs : np.ndarray
        Array with (n_steps, channels), being the n_steps the days of avaiable data and
        channels the number of variables used as target.
    times : pd.DatetimeIndex
        Index that contains the dates of the data.
        
    '''
    
    df = pd.read_csv(path, index_col=0, parse_dates=['datetime'])
    
    df['e'] = e(df['tdew'])
    df['u'] = df['wspd'] * -np.sin(np.radians(df['wdir']))
    df['v'] = df['wspd'] * -np.cos(np.radians(df['wdir']))
    
    u_daily = df['u'].resample('D').mean() # perde informação de horarios especificos e inversões
    v_daily = df['v'].resample('D').mean()
    
    prec_daily = df['prec'].resample('D').sum()
    tmax_daily = df['temp'].resample('D').max()
    tmin_daily = df['temp'].resample('D').min()
    tamp_daily = tmax_daily - tmin_daily
    emax_daily = df['e'].resample('D').max()
    emin_daily = df['e'].resample('D').min()
    
    yearly_cos = np.cos(2 * np.pi * np.clip((prec_daily.index.day_of_year - 1), 0, 364) / 365)
    yearly_sin = np.sin(2 * np.pi * np.clip((prec_daily.index.day_of_year - 1), 0, 364) / 365)
    
    inputs = np.stack([u_daily, v_daily, tmax_daily, tmin_daily, tamp_daily, emax_daily, emin_daily, yearly_cos, yearly_sin], axis=-1)
    outputs = prec_daily.values[:, None]
    times = prec_daily.index

    return inputs, outputs, times
    

def load_data_v1(path='data/sp_mirante.csv'):
    '''
    Load data from one .csv, take those variables and prepare train and test sets.
    
    Returns arrays with the information needed for training and testing,
    and additional information about the samples gathered.
    
    Parameters
    ----------
    path : str
        String that contains the path to the .csv file to be used.
        
    Returns
    -------
    x_train : np.ndarray
        Array with samples of the input data (samples, 10, 9) for
        training
    [y_train, y_train_bin] : list
        List of np.ndarray's with the target data (samples, 13, 1) and
        (samples, 13, 2) for training
    x_test : np.ndarray
        Array with samples of the input data (samples, 10, 9) for
        testing
    [y_test, y_test_bin] : list
        List of np.ndarray's with the target data (samples, 13, 1) and
        (samples, 13, 2) for testing
    weights : list
        List of weights for balancing classes
    times_train : np.ndarray
        Array with head time of each sample of the training set
    times_test : np.ndarray
        Array with head time of each sample of the testing set
    
    '''
    
    inputs, outputs, times = load_base_v1(path)
    
    train_mask = (times >= '2007-01') & (times <= '2020-12')
    test_mask = times >= '2021-01'
    
    data_train = (inputs[train_mask], outputs[train_mask])
    data_test = (inputs[test_mask], outputs[test_mask])
    times_train, times_test = times[train_mask], times[test_mask]
    times_train = times_train[:-12]
    times_test = times_test[:-12]
    
    x_train = []
    y_train = []
    for i in range(data_train[0].shape[0] - 9 - 3):
        in_slice = slice(i, i + 10)
        out_slice = slice(i, i + 10 + 3)
        x_train.append(data_train[0][in_slice])
        y_train.append(data_train[1][out_slice])
    
    x_test = []
    y_test = []
    for i in range(data_test[0].shape[0] - 9 - 3):
        in_slice = slice(i, i + 10)
        out_slice = slice(i, i + 10 + 3)
        x_test.append(data_test[0][in_slice])
        y_test.append(data_test[1][out_slice])
    
    x_train, y_train, x_test, y_test = list(map(np.array, [x_train, y_train, x_test, y_test]))
    
    # filter nan
    valid_train = np.min(~np.isnan(x_train), axis=(1, 2))
    x_train = x_train[valid_train]
    y_train = y_train[valid_train]
    times_train = times_train[valid_train]
    
    valid_test = np.min(~np.isnan(x_test), axis=(1, 2))
    x_test = x_test[valid_test]
    y_test = y_test[valid_test]
    times_test = times_test[valid_test]
    
    # bin & log
    y_train_bin = np.int32(y_train > 1) # > 1mm
    y_test_bin = np.int32(y_test > 1)
    
    y_train = np.log(y_train + 1)
    y_test = np.log(y_test + 1)
    
    weights = [
        y_train_bin.shape[0] / (2 * np.sum(y_train_bin == 0)),
        y_train_bin.shape[0] / (2 * np.sum(y_train_bin == 1)),
    ]
    
    return x_train, [y_train, y_train_bin], x_test, [y_test, y_test_bin], weights, times_train, times_test
    
    
def load_base_v2(path):
    '''
    D*
    
    Returns inputs, outputs and times, all prepared with channels, but not sample
    dimension
    
    Parameters
    ----------
    path : str
        String that contains the path to the .csv file.
        
    Returns
    -------
    inputs : np.ndarray
        Array with (n_steps, channels), being the n_steps the days of avaiable data and
        channels the number of variables used as input.
    outputs : np.ndarray
        Array with (n_steps, channels), being the n_steps the days of avaiable data and
        channels the number of variables used as target.
    times : pd.DatetimeIndex
        Index that contains the dates of the data.
        
    '''
    
    df = pd.read_csv(path, index_col=0, parse_dates=['datetime'])
    
    df['e'] = e(df['tdew'])
    
    prec_daily = df['prec'].resample('D').sum()

    tmax_daily = df['temp'].resample('D').max()
    tmin_daily = df['temp'].resample('D').min()
    tmean_daily = df['temp'].resample('D').mean()
    tamp_daily = tmax_daily - tmin_daily
    
    emax_daily = df['e'].resample('D').max()
    emin_daily = df['e'].resample('D').min()
    emean_daily = df['e'].resample('D').mean()
    
    yearly_cos = np.cos(2 * np.pi * np.clip((prec_daily.index.day_of_year - 1), 0, 364) / 365)
    yearly_sin = np.sin(2 * np.pi * np.clip((prec_daily.index.day_of_year - 1), 0, 364) / 365)
    
    inputs = np.stack([tmax_daily, tmin_daily, tmean_daily, tamp_daily, emax_daily, emin_daily, emean_daily, yearly_cos, yearly_sin], axis=-1)
    outputs = prec_daily.values[:, None]
    times = prec_daily.index

    return inputs, outputs, times


def load_data_v2(path='data/sp_mirante.csv'):
    '''
    Load data from one .csv, take those variables and prepare train and test sets.
    
    Returns arrays with the information needed for training and testing,
    and additional information about the samples gathered.
    
    Parameters
    ----------
    path : str
        String that contains the path to the .csv file to be used.
        
    Returns
    -------
    x_train : np.ndarray
        Array with samples of the input data (samples, 10, 9) for
        training
    [y_train, y_train_bin] : list
        List of np.ndarray's with the target data (samples, 13, 1) and
        (samples, 13, 2) for training
    x_test : np.ndarray
        Array with samples of the input data (samples, 10, 9) for
        testing
    [y_test, y_test_bin] : list
        List of np.ndarray's with the target data (samples, 13, 1) and
        (samples, 13, 2) for testing
    weights : list
        List of weights for balancing classes
    times_train : np.ndarray
        Array with head time of each sample of the training set
    times_test : np.ndarray
        Array with head time of each sample of the testing set
    
    '''
    
    inputs, outputs, times = load_base_v2(path)
    
    train_mask = (times >= '2007-01') & (times <= '2020-12')
    test_mask = times >= '2021-01'
    
    data_train = (inputs[train_mask], outputs[train_mask])
    data_test = (inputs[test_mask], outputs[test_mask])
    times_train, times_test = times[train_mask], times[test_mask]
    times_train = times_train[:-12]
    times_test = times_test[:-12]
    
    x_train = []
    y_train = []
    for i in range(data_train[0].shape[0] - 9 - 3):
        in_slice = slice(i, i + 10)
        out_slice = slice(i, i + 10 + 3)
        x_train.append(data_train[0][in_slice])
        y_train.append(data_train[1][out_slice])
    
    x_test = []
    y_test = []
    for i in range(data_test[0].shape[0] - 9 - 3):
        in_slice = slice(i, i + 10)
        out_slice = slice(i, i + 10 + 3)
        x_test.append(data_test[0][in_slice])
        y_test.append(data_test[1][out_slice])
    
    x_train, y_train, x_test, y_test = list(map(np.array, [x_train, y_train, x_test, y_test]))
    
    # filter nan
    valid_train = np.min(~np.isnan(x_train), axis=(1, 2))
    x_train = x_train[valid_train]
    y_train = y_train[valid_train]
    times_train = times_train[valid_train]
    
    valid_test = np.min(~np.isnan(x_test), axis=(1, 2))
    x_test = x_test[valid_test]
    y_test = y_test[valid_test]
    times_test = times_test[valid_test]
    
    # bin & log
    y_train_bin = np.int32(y_train > 1) # > 1mm
    y_test_bin = np.int32(y_test > 1)
    
    y_train = np.log(y_train + 1)
    y_test = np.log(y_test + 1)
    
    weights = [
        y_train_bin.shape[0] / (2 * np.sum(y_train_bin == 0)),
        y_train_bin.shape[0] / (2 * np.sum(y_train_bin == 1)),
    ]
    
    return x_train, [y_train, y_train_bin], x_test, [y_test, y_test_bin], weights, times_train, times_test


def load_base_v3(path):
    '''
    D*
    
    Returns inputs, outputs and times, all prepared with channels, but not sample
    dimension
    
    Parameters
    ----------
    path : str
        String that contains the path to the .csv file.
        
    Returns
    -------
    inputs : np.ndarray
        Array with (n_steps, channels), being the n_steps the days of avaiable data and
        channels the number of variables used as input.
    outputs : np.ndarray
        Array with (n_steps, channels), being the n_steps the days of avaiable data and
        channels the number of variables used as target.
    times : pd.DatetimeIndex
        Index that contains the dates of the data.
        
    '''
    
    df = pd.read_csv(path, index_col=0, parse_dates=['datetime'])
    
    df['e'] = e(df['tdew'])
    
    prec_daily = df['prec'].resample('D').sum()

    tmax_daily = df['temp'].resample('D').max()
    tmin_daily = df['temp'].resample('D').min()
    tmean_daily = df['temp'].resample('D').mean()
    tamp_daily = tmax_daily - tmin_daily
    
    emax_daily = df['e'].resample('D').max()
    emin_daily = df['e'].resample('D').min()
    emean_daily = df['e'].resample('D').mean()
    
    yearly_cos = np.cos(2 * np.pi * np.clip((prec_daily.index.day_of_year - 1), 0, 364) / 365)
    yearly_sin = np.sin(2 * np.pi * np.clip((prec_daily.index.day_of_year - 1), 0, 364) / 365)
    
    inputs = np.stack([tmax_daily, tmin_daily, tmean_daily, tamp_daily, emax_daily, emin_daily, emean_daily, yearly_cos, yearly_sin], axis=-1)
    outputs = prec_daily.values[:, None]
    times = prec_daily.index

    return inputs, outputs, times


def load_era5(path, clat, clon):
    '''
    Load era5 data from .nc files. Syncing the existing times into one
    single grid.
    
    Returns grid, lats and lons that will be used to load and interpolate
    goes data and times of each grid step.
    
    Parameters
    ----------
    path : str
        placeholder for a string with the path to the era5 data
    clat : float
        central float to the station latitude coordinate.
    clon : float
        central float to the station longitude coordinate.
        
    Returns
    -------
    : np.ndarray
        Concatenation of the grids, on the channel dimension.
    target_lats : np.ndarray
        Latitudes where goes will be interpolated.
    target_lons : np.ndarray
        Longitudes where goes will be interpolated.
    : pd.DatetimeIndex
        Index containing the times where the data was gathered.
        
    '''
    
    vars = ['T', 'Td', 'msl']
    year = 2018 # use only 2018 - goes has only this one xD
    
    time = None
    islice = None
    jslice = None
    target_lats = None
    target_lons = None
    
    grids = []
    for var in vars:
        idx = 12 # index where year 2018 is stored
        nc_file = nc.Dataset(f'D:\\era5\\{var}_%d.nc'%idx)
        load_var = list(set(nc_file.variables).difference(set(nc_file.dimensions)))[0]
        
        if time is None:
            time = ncHelper.load_time(nc_file['time'])
            lats, lons = ncHelper.get_lats_lons(nc_file)
            i = np.argmin(np.abs(lats - clat))
            j = np.argmin(np.abs(lons - clon))
            
            islice = slice(max(0, i - 90), i + 91)
            jslice = slice(max(0, j - 90), j + 91)
            
            target_lats = lats[islice][::-1]
            target_lons = lons[jslice]
        
        local_time = ncHelper.load_time(nc_file['time'])
        
        _, _, local_indices = np.intersect1d(time, local_time, return_indices=True)
        
        grid = nc_file[load_var][local_indices, islice, jslice][:, ::-1, :, None]
        print(grid.shape)
        grids.append(grid)
        
    return np.concatenate(grids, axis=-1), target_lats, target_lons, pd.Index(pd.to_datetime(time, utc=True))


def load_goes(path, tlats, tlons):
    '''
    Load goes data from .nc files. Using tlats and tlons to interpolate
    the grid of each file to the preferred dimension.
    
    Returns grid and time of valid found datetimes.
    
    Parameters
    ----------
    path : str
        placeholder for a string with the path to the era5 data
    tlats : float
        Latitudes used in the interpolation.
    tlons : float
        Longitudes used in the interpolation.
        
    Returns
    -------
    : np.ndarray
        Concatenation of the grids, on the channel dimension.
    : pd.DatetimeIndex
        Index containing the times where the data was gathered.
        
    '''
    
    base = 'D:\\cptec\\S10635346_2018%m%d%H00.nc'
    datetimes = pd.to_datetime(pd.date_range('2018-01-01T00', '2018-12-31T23', freq='6H'), utc=True)
    
    valid_times = []
    grids = []
    for datetime in tqdm(datetimes):
        if glob.glob(datetime.strftime(base)) != []:
            nc_file = nc.Dataset(datetime.strftime(base))
            lats, lons = ncHelper.get_lats_lons(nc_file)
            
            grids.append(RectBivariateSpline(lats, lons, nc_file['Band1'][:])(tlats, tlons)[None, ..., None])
            valid_times.append(datetime)
        
    return np.concatenate(grids, axis=0), pd.Index(valid_times)
        


def load_data_v3(path='data/sp_mirante.csv', era5_path='', goes_path=''):
    '''
    Load data from one .csv, take those variables and prepare train and test sets.
    
    Returns arrays with the information needed for training and testing,
    and additional information about the samples gathered.
    
    Parameters
    ----------
    path : str
        String that contains the path to the .csv file to be used.
        
    Returns
    -------
    x_train : np.ndarray
        Array with samples of the input data (samples, 10, 9) for
        training
    [y_train, y_train_bin] : list
        List of np.ndarray's with the target data (samples, 13, 1) and
        (samples, 13, 2) for training
    x_test : np.ndarray
        Array with samples of the input data (samples, 10, 9) for
        testing
    [y_test, y_test_bin] : list
        List of np.ndarray's with the target data (samples, 13, 1) and
        (samples, 13, 2) for testing
    weights : list
        List of weights for balancing classes
    times_train : np.ndarray
        Array with head time of each sample of the training set
    times_test : np.ndarray
        Array with head time of each sample of the testing set
    
    '''
    
    inputs, outputs, times = load_base_v3(path)
    
    center_lat, center_lon = -23.48, -46.63
    era5_inputs, target_lats, target_lons, era5_times = load_era5(era5_path, center_lat, center_lon) # get (N, 181, 181, c) grid
    goes_inputs, goes_times = load_goes(goes_path, target_lats, target_lons) # usage of RectBivariateSpline & (M, 181, 181, 1) grid
    
    train_mask = (times >= '2018-01') & (times <= '2018-11')
    test_mask = times >= '2018-12'
    
    data_train = (inputs[train_mask], outputs[train_mask])
    data_test = (inputs[test_mask], outputs[test_mask])
    times_train, times_test = times[train_mask], times[test_mask]
    times_train = times_train[:-12]
    times_test = times_test[:-12]
    
    x_train = []
    era5_train = []
    goes_train = []
    y_train = []
    for i in range(data_train[0].shape[0] - 9 - 3):
        in_slice = slice(i, i + 10)
        out_slice = slice(i, i + 10 + 3)
        x_train.append(data_train[0][in_slice])
        y_train.append(data_train[1][out_slice])
        
        it = np.argmin(np.abs(era5_times - times_train[in_slice][-1]))
        era5_train.append(era5_inputs[it])
        
        it = np.argmin(np.abs(goes_times - times_train[in_slice][-1]))
        goes_train.append(goes_inputs[it])
    
    x_test = []
    era5_test = []
    goes_test = []
    y_test = []
    for i in range(data_test[0].shape[0] - 9 - 3):
        in_slice = slice(i, i + 10)
        out_slice = slice(i, i + 10 + 3)
        x_test.append(data_test[0][in_slice])
        y_test.append(data_test[1][out_slice])
        
        it = np.argmin(np.abs(era5_times - times_test[in_slice][-1]))
        era5_test.append(era5_inputs[it])
        
        it = np.argmin(np.abs(goes_times - times_test[in_slice][-1]))
        goes_test.append(goes_inputs[it])
    
    x_train, y_train, x_test, y_test = list(map(np.array, [x_train, y_train, x_test, y_test]))
    era5_train, goes_train, era5_test, goes_test = list(map(np.array, [era5_train, goes_train, era5_test, goes_test]))
    
    # filter nan
    valid_train = np.min(~np.isnan(x_train), axis=(1, 2))
    x_train = x_train[valid_train]
    y_train = y_train[valid_train]
    times_train = times_train[valid_train]
    era5_train = era5_train[valid_train]
    goes_train = goes_train[valid_train]
    
    valid_test = np.min(~np.isnan(x_test), axis=(1, 2))
    x_test = x_test[valid_test]
    y_test = y_test[valid_test]
    times_test = times_test[valid_test]
    era5_test = era5_test[valid_test]
    goes_test = goes_test[valid_test]
    
    # bin & log
    y_train_bin = np.int32(y_train > 1) # > 1mm
    y_test_bin = np.int32(y_test > 1)
    
    y_train = np.log(y_train + 1)
    y_test = np.log(y_test + 1)
    
    weights = [
        y_train_bin.shape[0] / (2 * np.sum(y_train_bin == 0)),
        y_train_bin.shape[0] / (2 * np.sum(y_train_bin == 1)),
    ]
    
    return [x_train, era5_train, goes_train], [y_train, y_train_bin], [x_test, era5_test, goes_test], [y_test, y_test_bin], weights, times_train, times_test
    

if __name__ == '__main__': # testing
    res = load_data_v3('../data/sp_mirante.csv')
    
    print(res)