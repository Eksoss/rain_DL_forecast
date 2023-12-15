import numpy as np
import pandas as pd

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
    
    u_daily = df['u'].resample('D').mean()
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
    
    data_train = (inputs[:-700], outputs[:-700])
    data_test = (inputs[-700:], outputs[-700:])
    times_train, times_test = times[:-700], times[-700:]
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