#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:38:46 2021

@author: mattwear
"""

import pandas as pd
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
    df = pd.concat([df, pd.DataFrame(tfidf)], axis = 1)
    return(df)


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
    
    #one hot encodings
    df = color_clean(df, color_list=['white','black','silver'])
    df = drive_clean(df)
    df = transmission_clean(df)
    df = cleanLocationFeatures(df)
    
    #impute some missing values
    
    
    #remove remaining missing values
    

    
    return(df)