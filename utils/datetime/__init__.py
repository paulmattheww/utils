import pandas as pd

def datetime_days_to_float(dat):
    return str(dat).split(" days")[0]

def derive_years(now, prior):
    """Retrive the distance between two dates.
    Must pass Series as datetime (use pd.datetime).

    ARGS:
        - now: Series of datetime
        - prior: Series of datetime
    """
    try:
        years = round((now - prior).dt.days / 365.25, 1)
    except AttributeError:
        years = round((now - prior) / 365.25, 1)

    return years

def generate_calendar(year, drop_index=False):
    '''
    Simple function to generate a calendar containing
    US holidays, weekdays and  holiday weeks.
    '''
    from pandas.tseries.offsets import YearEnd
    from pandas.tseries.holiday import USFederalHolidayCalendar

    start_date = pd.to_datetime('1/1/'+str(year))
    end_date = start_date + YearEnd()
    DAT = pd.date_range(str(start_date), str(end_date), freq='D')
    MO = [d.strftime('%B') for d in DAT]
    holidays = USFederalHolidayCalendar().holidays(start=start_date, end=end_date)

    cal_df = pd.DataFrame({'date':DAT, 'month':MO})
    cal_df['year'] = [format(d, '%Y') for d in DAT]
    cal_df['weekday'] = [format(d, '%A') for d in DAT]
    cal_df['is_weekday'] = cal_df.weekday.isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])
    cal_df['is_weekday'] = cal_df['is_weekday'].astype(int)
    cal_df['is_holiday'] = cal_df['date'].isin(holidays)
    cal_df['is_holiday'] = cal_df['is_holiday'].astype(int)
    cal_df['is_holiday_week'] = cal_df.is_holiday.rolling(window=7,center=True,min_periods=1).sum()
    cal_df['is_holiday_week'] = cal_df['is_holiday_week'].astype(int)

    if not drop_index: cal_df.set_index('date', inplace=True)

    return cal_df

def make_calendars(year_list, drop_index):
    """
    Example:
        year_list = ['2016', '2017', '2018']
        cal_df = make_calendars(year_list, drop_index=True)
        cal_df.head()
    """
    cal_df = pd.DataFrame()
    for year in year_list:
        cal_df = cal_df.append(generate_calendar(year, drop_index=drop_index))
    return cal_df


def force_datetime(dat):
    try:
        dat = pd.to_datetime(str(dat))
    except:
        dat = pd.to_datetime("NaT")
    return dat
