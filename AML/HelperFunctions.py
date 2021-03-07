#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:28:06 2021

@author: mattwear
"""

###M5 Helper Functions

import pandas as pd
import numpy as np
import os
import random

def meltM5(df, days = 1200, items = 1000):
    """Function returns melted version of M5 sales data

    Parameters
    ----------
    df : Pandas DataFrame
        M5 Sales data
    days : int
        How many days of sales to include (default is all 1941 days)
    items: int
        How many items to include (default is 30490)

    Returns
    -------
    df_melt
        Sales data in a melted format
    """
    
    #Random sample of items items
    df_melt = df.sample(n=items, random_state=30).copy().reset_index(drop=True)
    #Subset to include specified number of days
    df_melt = df_melt.iloc[:, :days+6]
    #Melt
    df_melt = pd.melt(df_melt, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                 var_name='d', value_name='sold')
    
    return df_melt

def joinDataSets(melt_df, calendar_df, prices_df, dropPriceNA=True):
    """Function joins melted dataframe with calendar and sell price dataframes

    Parameters
    ----------
    melt_df : Pandas DataFrame
        Melted form of original data set
    calendar_df : Pandas DataFrame
        Calendar dataframe
    prices_df : Pandas DataFrame
        sell prices dataframe
    dropPriceNA: boolean, optional
        drop rows for which a sell price is not given (since this means that the 
        product was not on sale that day), default is True

    Returns
    -------
    df_joined
        Combined all available data into one dataframe in melted format
    """
    #Join calendar data
    df_joined = pd.merge(melt_df, calendar_df, on='d', how='left')
    #Join sell prices data
    df_joined = pd.merge(df_joined, prices_df, on=['store_id','item_id','wm_yr_wk'], how='left') 
    #Drop rows where sell price is na
    if dropPriceNA:
        df_joined.dropna(subset=['sell_price'], inplace=True)
        
    return df_joined




def load_calendar_data():    
    csv_path = os.path.join('data', 'calendar.csv')
    return pd.read_csv(csv_path)


def load_sell_price_data():
    csv_path = os.path.join('data', 'sell_prices.csv')
    return pd.read_csv(csv_path)


def load_sales_train_validation_data():
    csv_path = os.path.join('data', 'sales_train_validation.csv')
    return pd.read_csv(csv_path)


def load_sales_train_evaluation_data():
    csv_path = os.path.join('data', 'sales_train_evaluation.csv')
    return pd.read_csv(csv_path)


def load_meltedJoined_data():    
    csv_path = os.path.join('data', 'meltedJoinedDf.csv')
    return pd.read_csv(csv_path)

def load_reduced_data():    
    csv_path = os.path.join('data', 'reduced.csv')
    return pd.read_csv(csv_path)



def reduceDataset(df, cuttOffDate = "2011-06-29", numberOfProducts = 500):
    df_reduced = df.loc[df["date"] < cuttOffDate]
    random_products = random.choices(list(set(df_reduced["id"])), k = numberOfProducts)
    return(df_reduced[df_reduced['id'].isin(random_products)])
    
    
def cleanDates(df):
    #Drops weekday, date and wm_yr_wk and creates a weekend binary indicator
    df.drop(['weekday','date','wm_yr_wk'], axis=1, inplace=True)
    #df = df.assign(weekend=lambda df: df.apply(lambda row: 1 if row.wday == 1 or row.wday == 2 else 0, axis=1))
    sat = df['wday'] == 1 
    sun = df['wday'] == 2
    mon = df['wday'] == 3
    tue = df['wday'] == 4
    wed = df['wday'] == 5
    thur = df['wday'] == 6
    fri = df['wday'] == 7
    df['weekend'] = (sat | sun).astype(int)
    df['midweek'] = (tue | wed | thur).astype(int)
    df['monfri'] = (mon | fri).astype(int)
    #Convert the d column into integer
    df['d'] = df['d'].str.replace('d_', '').astype(int)
    return df

def cleanEvents(df):
    #Converts NaN to 'nan'
    df['event_type_1'] = df['event_type_1'].astype(str)
    df['event_type_2'] = df['event_type_2'].astype(str)

    #Find the unique events in event_type_2
    nan_array = np.array(['nan'])
    unique_types = np.unique(df.event_type_2)
    type_loop = np.setdiff1d(unique_types,nan_array)

    #One-hot encoding event types
    dummy = pd.get_dummies(df[['event_type_1','event_type_2']])
    df = pd.concat([df, dummy], axis=1)
    for e in type_loop:
        df['event_type_1_'+e] = ((df['event_type_1_'+e] == 1)  | (df['event_type_2_'+e] == 1)).astype(int)

    #One-hot encode specific event names
    df['Christmas'] = ((df['event_name_1'] == 'Christmas')  | (df['event_name_2'] == 'Christmas')).astype(int)

    #Drop unneeded columns
    df.drop(['event_name_1','event_name_2','event_type_1','event_type_2'], axis=1, inplace=True)
    df.drop(list(df.filter(regex = 'event_type_2')), axis = 1, inplace = True)

    #Rename column names
    df.rename(columns={"event_type_1_National": "National", 
                       "event_type_1_Religious": "Religious",
                        "event_type_1_Cultural": "Cultural",
                      "event_type_1_Sporting": "Sporting",
                      "event_type_1_nan": "NoEvent"}, inplace=True)
    
    return df


def add_event_eve(df):
    #adds the eve of every event in the event_name_2 collumn
    #and the type of event in the type_of_event_2 column
    
    event_names=df['event_name_1'].unique().tolist()[1:]
    
    for i in range(df.shape[0]):
        if (df["event_name_1"][i] in event_names) and (df["event_name_2"][i-1] not in event_names):
            df['event_name_2'][i-1]= df["event_name_1"][i] + " Eve"
            df['event_type_2'][i-1]= df['event_type_1'][i]
                
    return df



def important_events_onehotenc(df):
    # A function for one-hot-encoding the events that may be important
    # SuperBowl, LaborDay, Easter Eve, Mother's day Eve, SuperBowl Eve, 
    # PresidentsDay Eve, Columbus day Eve, Thanksgiving and Lentweek2 Eve
    drop_events=['event_Chanukah End', 'event_Christmas','event_Cinco De Mayo', 'event_ColumbusDay', 'event_Easter',
                 'event_Eid al-Fitr', 'event_EidAlAdha', "event_Father's day",'event_Halloween', 'event_IndependenceDay',
                 'event_LentStart', 'event_LentWeek2', 'event_MartinLutherKingDay','event_MemorialDay', "event_Mother's day", 
                 'event_NBAFinalsEnd', 'event_NBAFinalsStart', 'event_NewYear', 'event_OrthodoxChristmas','event_OrthodoxEaster',
                 'event_Pesach End', 'event_PresidentsDay','event_Purim End', 'event_Ramadan starts', 'event_StPatricksDay',
                 'event_Thanksgiving', 'event_ValentinesDay','event_VeteransDay', 'event_Chanukah End Eve', 'event_Christmas Eve',
                 'event_Cinco De Mayo', 'event_Cinco De Mayo Eve', 'event_Easter','event_Eid al-Fitr Eve', 'event_EidAlAdha Eve',
                 "event_Father's day", "event_Father's day Eve", 'event_Halloween Eve','event_IndependenceDay Eve', 
                 'event_LaborDay Eve','event_LentStart Eve','event_MartinLutherKingDay Eve', 'event_MemorialDay Eve', 
                 'event_NBAFinalsEnd Eve','event_NBAFinalsStart Eve', 'event_NewYear Eve','event_OrthodoxChristmas Eve', 
                 'event_OrthodoxEaster','event_OrthodoxEaster Eve', 'event_Pesach End Eve', 'event_Purim End Eve',
                 'event_Ramadan starts Eve', 'event_StPatricksDay Eve','event_ValentinesDay Eve', 'event_VeteransDay Eve']
    df=pd.get_dummies(data=df,prefix="event",columns=["event_name_1","event_name_2"])
    df.drop(columns=drop_events,axis=1,inplace=True)
    
    return df
    


def encodeItemAndStoreData(df):
    df.drop(['item_id'], axis=1, inplace=True)

    encode_variables = ['dept_id', 'cat_id','store_id','state_id']
    for var in encode_variables:
        dummy = pd.get_dummies(df[var])
        df = pd.concat([df, dummy], axis=1)
        df.drop([var], axis=1, inplace=True)
        
    return df

def rollingMeanDemandFeature(data, windowSize, shift):
    data['rolling_mean_'+str(windowSize)+'_'+str(shift)] = data.groupby(['id'])['sold'].transform(lambda x: x.shift(shift).rolling(windowSize).mean())
    return data

def rollingStdDemandFeature(data, windowSize, shift):
    data['rolling_std_'+str(windowSize)+'_'+str(shift)] = data.groupby(['id'])['sold'].transform(lambda x: x.shift(shift).rolling(windowSize).std())
    return data

def rollingMeanWeekday(df, weeks, shift):
    
    """
    Inputs are dataframe with number of sales, number of weeks to average, and number to shift
    
    Takes the rolling mean average of the past n weeks for that particular weekday
    returns dataframe with added column of the rolling averages
    """
    
    data = pd.DataFrame(columns = df.columns)
    for day in range(7):
        d = df[day:][::7]
        d = rollingMeanDemandFeature(d, windowSize= weeks, shift=shift) 
        data = pd.concat([data, d])
    
    data.rename(columns = {"rolling_mean_{}_{}".format(str(weeks), str(shift)):\
                           "rolling_weeks_{}_{}".format(str(weeks), str(shift))})
    data = data.sort_index()
    return(data)


def lagFeature(df, var='sold', lag=1):
    df['sold_lag_'+str(lag)] = df.groupby(['id'])[var].shift(lag)
    return df

def priceDifference(df):
    #Real value of the price change
    df['price_diff'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.diff()) 
    df['price_diff'].fillna(0, inplace=True)
    #Binary variable indicating whether price increase/decreased that day
    df['price_increase'] = (df['price_diff'] > 0).astype(int)
    df['price_decrease'] = (df['price_diff'] < 0).astype(int)
    return df

def rollingPriceDifference(df, windowsize):
    #Has there been a price increase/decrease in the past windowsize days?
    #Can only be used if price_increase and price_decrease features are available as given in priceDifference(df)
    df['price_increase_'+str(windowsize)] = df.groupby(['id'])['price_increase'].transform(lambda x: x.rolling(windowsize).sum())
    df['price_increase_'+str(windowsize)] = (df['price_increase_'+str(windowsize)] > 0.5).astype(int)
    df['price_decrease_'+str(windowsize)] = df.groupby(['id'])['price_decrease'].transform(lambda x: x.rolling(windowsize).sum())
    df['price_decrease_'+str(windowsize)] = (df['price_decrease_'+str(windowsize)] > 0.5).astype(int)
    return df

def rawToClean(sales_df, calendar_df, price_df, days=300, items=100, dropNAPrices=True):
    df = meltM5(sales_df, days = days, items = items)
    df = joinDataSets(df, calendar_df, price_df, dropPriceNA=dropNAPrices)
    df = cleanEvents(df)
    df = cleanDates(df)
    df = encodeItemAndStoreData(df)
    
    return df
    
    
def getTargetAndFeatures(df):
    y = df.sold.values
    X_df = df.drop(['sold'], axis=1)
    X = X_df.values
    return y, X



# type casting to reduce memory usage
def typeCastCalendar(calendar):
    # month, wdays are small numbers and snap_CA, snap_TX, and snap_WI are just booleans
    calendar[["month", 'wday', "snap_CA", "snap_TX", "snap_WI"]]= calendar[["month", 'wday', "snap_CA", "snap_TX", "snap_WI"]].astype("int8")

    calendar[['wm_yr_wk', 'year']] = calendar[['wm_yr_wk', 'year']].astype('int16')

    calendar[['date']] = calendar[['date']].astype('datetime64')

    calendar[['weekday', "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = calendar[['weekday', "event_name_1", "event_type_1", "event_name_2", "event_type_2"]].astype('category')

    return calendar

def typeCastSales(data):
    data.iloc[:, 6:] = data.iloc[:, 6:].astype("int16")
    return data

def typeCastPrice(data):
    data[['wm_yr_wk']] = data[['wm_yr_wk']].astype('int16')
    data["sell_price"] = data["sell_price"].astype("float16")
    return data
    