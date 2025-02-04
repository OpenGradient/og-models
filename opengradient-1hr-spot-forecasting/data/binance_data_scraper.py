import requests
from io import BytesIO
import zipfile
import pandas as pd
from time import sleep

def download_binance_candles_file(
        pair,
        date,
        range_size='monthly',
        asset_type='spot',
        granularity='1m',
        save_directory='./',
        make_datetime=False,
        pause_time=0):
    """
    Download a zip file of Binance kline data for a given symbol, date, 
    and resolution, then return or save the CSV.

    Parameters
    ----------
    pair : str
        Binance format specified pair, e.g. 'ETHUSDT'.
    date : str
        If 'range_size' is 'monthly' then a month (e.g. '2024-01'), 
        if 'range_size' is 'daily' then a day (e.g. '2024-01-01').
    range_size : str
        'monthly' or 'daily'; default is 'monthly'.
    asset_type : str
        'spot', 'option', or 'futures'; default is 'spot'.
    granularity : str
        '1s', '1m', '1h', etc.; default is '1m'.
    save_directory : str
        Directory to save CSV. If None, returns a pandas DataFrame instead.
    make_datetime : bool
        Whether to parse timestamp_ms as a datetime column.
    pause_time : float
        Sleep time between requests to avoid rate limits.

    Returns
    -------
    If save_directory is None, returns a pandas DataFrame.
    Otherwise saves CSV and returns None.
    """
    column_names = [
        'timestamp_ms',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_volume',
        'count',
        'taker_buy_volume',
        'taker_buy_quote_volume',
        'ignore'
    ]
    
    zipname = f"{pair}-{granularity}-{date}.zip"
    csv_name = zipname.replace('.zip', '.csv')
    url = (
        f"https://data.binance.vision/data/{asset_type}/{range_size}/klines/"
        f"{pair}/{granularity}/{zipname}"
    )

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"Failed to download: {url} (status code: {r.status_code})")
        return None

    z = zipfile.ZipFile(BytesIO(r.content))    
    df = pd.read_csv(z.open(csv_name), names=column_names)

    if make_datetime:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

    sleep(pause_time)

    if save_directory is None:
        return df
    else:
        save_path = f"{save_directory.rstrip('/')}/{csv_name}"
        df.to_csv(save_path, index=False)
        print(f"Saved {save_path}")

def download_binance_candles_daterange(
        pair,
        start_date,
        end_date,
        asset_type='spot',
        granularity='1m',
        save_directory='./',
        save_filename=None,
        pause_time=0.1,
        make_datetime=False):
    """
    Download Binance kline data between a start_date and end_date 
    (daily or monthly resolution) and save/return a consolidated DataFrame.

    Parameters
    ----------
    pair : str
        E.g. 'ETHUSDT'.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    asset_type : str
        'spot', 'option', or 'futures'.
    granularity : str
        '1s', '1m', '1h', etc.
    save_directory : str
        Directory for saving the final CSV. If None, returns DataFrame.
    save_filename : str
        If provided, the final CSV will have this name.
    pause_time : float
        Sleep time between file downloads.
    make_datetime : bool
        Whether to parse timestamp_ms as a datetime column.

    Returns
    -------
    df or None
        If save_directory is None, a DataFrame is returned.
        Otherwise, CSV is written and None is returned.
    """
    start_date_datetime = pd.to_datetime(start_date)
    end_date_datetime = pd.to_datetime(end_date)

    # We attempt to download monthly data if range >= ~1 month 
    # and daily data for smaller spans or partial months.
    load_months = False
    if (end_date_datetime - start_date_datetime).days >= 28:
        # Heuristic to check if valid monthly range
        if start_date_datetime + pd.offsets.MonthEnd(0) + pd.DateOffset(months=1) <= end_date_datetime:
            load_months = True
        elif (start_date_datetime.day == 1 and 
              end_date_datetime == start_date_datetime + pd.offsets.MonthEnd(0)):
            # Perfect monthly block
            date_str = start_date_datetime.strftime('%Y-%m')
            df = download_binance_candles_file(
                pair=pair,
                date=date_str,
                range_size='monthly',
                granularity=granularity,
                save_directory=None,
                pause_time=pause_time,
                make_datetime=make_datetime
            )
            if save_directory is None:
                return df
            else:
                if save_filename is None:
                    csv_name = f"{pair}-{granularity}-{start_date}--{end_date}.csv"
                else:
                    csv_name = save_filename
                df.to_csv(f"{save_directory.rstrip('/')}/{csv_name}", index=False)
                return None

    if load_months:
        # Possibly partial daily intervals at start or end
        if start_date_datetime.day == 1:
            daily_datelist_begin = []
        else:
            daily_datelist_begin = pd.date_range(
                start_date,
                start_date_datetime + pd.offsets.MonthEnd(0),
                freq='d'
            )

        if end_date_datetime == end_date_datetime + pd.offsets.MonthEnd(0):
            daily_datelist_end = []
        else:
            daily_datelist_end = pd.date_range(
                end_date_datetime + pd.offsets.MonthBegin(0) - pd.DateOffset(months=1),
                end_date,
                freq='d'
            )

        if len(daily_datelist_begin) == 0:
            monthly_start = start_date
        else:
            monthly_start = start_date_datetime + pd.offsets.MonthBegin(0)

        if len(daily_datelist_end) == 0:
            monthly_end = end_date_datetime + pd.offsets.MonthBegin(0) - pd.DateOffset(months=1)
        else:
            monthly_end = end_date_datetime + pd.offsets.MonthBegin(0) - pd.DateOffset(months=2)

        monthly_datelist = pd.date_range(monthly_start, monthly_end, freq=pd.DateOffset(months=1))

        df_list = []
        for dt in daily_datelist_begin:
            date_str = dt.strftime('%Y-%m-%d')
            df_list.append(
                download_binance_candles_file(
                    pair=pair,
                    date=date_str,
                    range_size='daily',
                    granularity=granularity,
                    save_directory=None,
                    pause_time=pause_time,
                    make_datetime=make_datetime
                )
            )
        for dt in monthly_datelist:
            date_str = dt.strftime('%Y-%m')
            df_list.append(
                download_binance_candles_file(
                    pair=pair,
                    date=date_str,
                    range_size='monthly',
                    granularity=granularity,
                    save_directory=None,
                    pause_time=pause_time,
                    make_datetime=make_datetime
                )
            )
        for dt in daily_datelist_end:
            date_str = dt.strftime('%Y-%m-%d')
            df_list.append(
                download_binance_candles_file(
                    pair=pair,
                    date=date_str,
                    range_size='daily',
                    granularity=granularity,
                    save_directory=None,
                    pause_time=pause_time,
                    make_datetime=make_datetime
                )
            )

        df = pd.concat(df_list)
        df.drop_duplicates(subset=['timestamp_ms'], inplace=True)

    else:
        # Just daily
        df_list = []
        for dt in pd.date_range(start_date, end_date, freq='d'):
            date_str = dt.strftime('%Y-%m-%d')
            df_list.append(
                download_binance_candles_file(
                    pair=pair,
                    date=date_str,
                    range_size='daily',
                    granularity=granularity,
                    save_directory=None,
                    pause_time=pause_time,
                    make_datetime=make_datetime
                )
            )
        df = pd.concat(df_list)

    df.drop_duplicates(subset=['timestamp_ms'], inplace=True)
    df.sort_values(by='timestamp_ms', inplace=True)

    if save_directory is None:
        return df
    else:
        if save_filename is None:
            csv_name = f"{pair}-{granularity}-{start_date}--{end_date}.csv"
        else:
            csv_name = save_filename
        save_path = f"{save_directory.rstrip('/')}/{csv_name}"
        df.to_csv(save_path, index=False)
        print(f"Saved {save_path}")
        return None 