#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:38:46 2021

@author: mattwear
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

### Data Cleaning Functions

# Getting rid of unecesarry columns

def remove_columns(df):
    
    """Removes columns we agreed were useless"""
    columns = ["Unnamed: 0", "id", "image_url", "VIN", "region_url", "id", "model", "size","url"]
    df.drop(columns, axis=1, inplace=True)
    return(df)


def price_range(df, lower = 0, higher = 60_000, sampling = False):
    """
    Set the lower and upper limits of the price, if sampling true it chooses 40,000 samples at random
    """
    if sampling == True:
        df = df.sample(n = 40_000)
    
    df = df.dropna(subset = ["price"])
    
    df = df.loc[df["price"] < higher]
    df = df.loc[df["price"] >= lower]
    
    return(df)

def dateToDatetime(df):
    df['posting_date'] = df['posting_date'].str[:10].astype('datetime64[ns]')
    return df

def basicImpute(df):
    #Here, impute missing values with a common number/value
    
    ###Description: Impute missing values as empty string
    ###posting_date: Not many so impute with mean posting date
    mean_posting_date = np.mean(df.posting_date)

    #Fill NA
    df.fillna(value={'description': '', 'posting_date':mean_posting_date}, inplace=True)
    
    return df


def imputeMissingByManufacturer(df, col):
    ###Impute missing values by taking the most common occurence of the items manufacturer
    #Use this for fuel and transmission. The remaining missing values have NaN manufacturer
    
    cars = df.copy()
    cars_unique = np.unique(cars.manufacturer.astype(str))
    man_dict = {}

    for car in cars_unique:
        if car != 'nan':
            max_occur = cars[cars.manufacturer==car][col].value_counts().index[0]
            man_dict[car] = max_occur

    cars.loc[cars[col].isnull(),col] = cars['manufacturer'].map(man_dict)
    return cars


def TF_IDF(df, number = 100):
    
    """This function adds the TF-IDF values of the most important words to the dataframe,
    the number can be chosen above"""
    
    """TF - term frequency """
    
    """Inverse Document Frequency (IDF)
    IDF is a measure of how important a term is"""
    
    vectorizer = TfidfVectorizer(stop_words='english',max_features = number)
    sentences  = df["description"].values
    vectorizer.fit(sentences)
    vector_spaces = vectorizer.transform(sentences)
    tfidf = vector_spaces.toarray()
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, pd.DataFrame(tfidf)], axis = 1)
    return(df)


def manuf_country(cars):
    
    # Manufacturer country assigned
    
    cars.loc[ cars["manufacturer"] == "ford", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "chevrolet", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "toyota", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "honda", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "nissan", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "jeep", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "ram", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "gmc", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "dodge", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "bmw", "manuf_country"] = "Germany"
    cars.loc[ cars["manufacturer"] == "hyundai", "manuf_country"] = "S.Korea"
    cars.loc[ cars["manufacturer"] == "mercedes-benz", "manuf_country"] = "Germany"
    cars.loc[ cars["manufacturer"] == "subaru", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "volkswagen", "manuf_country"] = "Germany"
    cars.loc[ cars["manufacturer"] == "kia", "manuf_country"] = "S.Korea"
    cars.loc[ cars["manufacturer"] == "chrysler", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "lexus", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "cadillac", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "buick", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "mazda", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "audi", "manuf_country"] = "Germany"
    cars.loc[ cars["manufacturer"] == "acura", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "infiniti", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "lincoln", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "pontiac", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "volvo", "manuf_country"] = "Sweden"
    cars.loc[ cars["manufacturer"] == "mini", "manuf_country"] = "UK"
    cars.loc[ cars["manufacturer"] == "mitsubishi", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "porsche", "manuf_country"] = "Germany"
    cars.loc[ cars["manufacturer"] == "rover", "manuf_country"] = "UK"
    cars.loc[ cars["manufacturer"] == "mercury", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "saturn", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "tesla", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "jaguar", "manuf_country"] = "UK"
    cars.loc[ cars["manufacturer"] == "fiat", "manuf_country"] = "Italy"
    cars.loc[ cars["manufacturer"] == "alfa-romeo", "manuf_country"] = "Italy"
    cars.loc[ cars["manufacturer"] == "harley-davidson", "manuf_country"] = "USA"
    cars.loc[ cars["manufacturer"] == "ferrari", "manuf_country"] = "Italy"
    cars.loc[ cars["manufacturer"] == "datsun", "manuf_country"] = "Japan"
    cars.loc[ cars["manufacturer"] == "aston-martin", "manuf_country"] = "UK"
    cars.loc[ cars["manufacturer"] == "land-rover", "manuf_country"] = "UK"
    cars.loc[ cars["manufacturer"] == "morgan", "manuf_country"] = "UK"
    cars.loc[ cars["manufacturer"] == "hennessey", "manuf_country"] = "USA"
    
    # One hot encoding to one of USA, Japan, Germany, S.Korea, UK, Sweden, Italy
    
    cars = pd.get_dummies( cars, columns = ["manuf_country"])
    
    return cars


def ohe_type(cars):
    
    # Some condition groupings
    
    cars.loc[ cars["type"] == "mini-van", "type"] = "van/mini-van"
    cars.loc[ cars["type"] == "van", "type"] = "van/mini-van"
    cars.loc[ cars["type"] == "truck", "type"] = "pickup_truck"
    cars.loc[ cars["type"] == "pickup", "type"] = "pickup_truck"
    
    # One hot encoding type (original values)
    
    cars = pd.get_dummies( cars, columns = ["type"])
    
    return cars


def ohe_condition(cars):
    
    # One hot encoding condition (original values)
    
    cars = pd.get_dummies( cars, columns = ["condition"])
    
    return cars


def ohe_cylinders(cars):
    
    # One hot encoding cylinders (original values)
    
    cars = pd.get_dummies( cars, columns = ["cylinders"])
    
    return cars


def ohe_fuel(cars):

    # One hot encoding fuel (original values)
    
    cars = pd.get_dummies( cars, columns = ["fuel"])


def color_clean(df, color_list=['white','black','silver']):

    #groups all the colors that are not in the list as "other"
    #one hot encoding of paint_color column
    
    df["paint_color"]=df["paint_color"].apply(lambda x: x if x in color_list else "other")
    df=pd.get_dummies(df, prefix="color",columns=['paint_color'])
 
    return df



def drive_clean(df):
    
    #Assigns 4wd to all SUVs, pickups and offroads with nan drive type 
    df.loc[(((df["type"]=="SUV") | 
            (df["type"]=="pickup") | 
            (df["type"]=="offroad")) & (df['drive'].isnull()==True)),"drive"] = "4wd"
    
    #assign "other" to all nan values
    df.loc[(df['drive'].isnull()==True),"drive"]="other"

    #one hot encoding 4wd, rwd, fwd, other
    df = pd.get_dummies(df,prefix="drive",columns=['drive'])
    
    return df    


def transmission_clean(df):
    
    #Groups nan values with "other" type of transmission
    df.loc[(df['transmission'].isnull()==True),"transmission"]="other"
    
    #one hot encoding manual, automatic and other
    df = pd.get_dummies(df,prefix="transmission",columns=['transmission'])
    
    return df


def titlestatus_clean(df):
    df = pd.get_dummies(df,prefix="status",columns=['title_status'])
    return df


def fillLatLongNA(df):
    #Fills all missing lat, long values with the median of their respective region
    region_coords_ave_lat = df[['region','lat']].groupby(['region'])['lat'].median().to_dict()
    region_coords_ave_long = df[['region','long']].groupby(['region'])['long'].median().to_dict()
    df.loc[df['lat'].isnull(),'lat'] = df['region'].map(region_coords_ave_lat)
    df.loc[df['long'].isnull(),'long'] = df['region'].map(region_coords_ave_long)
    return df

def fillLatLongOutliers(df):
    latlong_outliers = (df.lat<20)  | (df.lat>70) | (df.long < -160) | (df.long > -60)
    region_coords_ave_lat = df[['region','lat']].groupby(['region'])['lat'].median().to_dict()
    region_coords_ave_long = df[['region','long']].groupby(['region'])['long'].median().to_dict()
    df.loc[latlong_outliers,'lat'] = df['region'].map(region_coords_ave_lat)
    df.loc[latlong_outliers,'long'] = df['region'].map(region_coords_ave_long)
    return df

def cleanLatLong(df):
    df = fillLatLongNA(df)
    df = fillLatLongOutliers(df)
    return df

def cleanLocationFeatures(df):
    df = cleanLatLong(df)
    #one hot encoding state
    df = pd.get_dummies(df,prefix="state",columns=['state'])
    #drop region
    df.drop(['region'], axis=1, inplace=True)
    
    return df  

def groupStateByPrice(df, lower=10000, higher=15000):
    ### Adds binary feature indicating if car was sold in cheap, medium or expensive state
    
    cars = df.copy()
    #Find median price by state
    bar = cars[['price','state']].groupby(['state'])['price'].median()
    bar.sort_values(ascending=False, inplace=True)
    
    #Find expensive, mid price and cheap states according to thresholds
    expensive_states = np.array(bar[bar >higher].index)
    mediumprice_state = np.array(bar[(bar > lower) & (bar <= higher)].index)
    cheap_state = np.array(bar[bar <= lower].index)
    
    #Add on hot encoded features to dataset
    cars['state_expensive'] = cars.state.isin(expensive_states).astype(int)
    cars['state_medium'] = cars.state.isin(mediumprice_state).astype(int)
    cars['state_cheap'] = cars.state.isin(cheap_state).astype(int)
    
    return cars

def ultimateClean(df):
    #remove useless values
    df = remove_columns(df)
    df = price_range(df, lower = 50, higher = 60_000, sampling = False)
    print("Cleaned !")
    
    #one hot encodings
    df = color_clean(df, color_list=['white','black','silver'])
    df = drive_clean(df)
    df = transmission_clean(df)
    df = titlestatus_clean(df)
    df = cleanLocationFeatures(df)
    
    df = manuf_country(df)
    df = ohe_condition(df)
    df = ohe_type(df)
    df = ohe_cylinders(df)
    df = ohe_fuel(df)
    
    print("One hot encodings done!")
    
    #impute some missing values
    
    
    #remove remaining missing values
    df = df.dropna()
    print("Dropped NANs!")

    
    return(df)


def normalise(df):
    cols_to_norm = ["year", "odometer", "lat", "long"]
    df[cols_to_norm] = MinMaxScaler().fit_transform(df[cols_to_norm])
    #FOR TRAIN TEST SPLIT USE SKLEARN train_test_split
    return(df)
