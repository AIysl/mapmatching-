import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
from datetime import datetime
import time
from CutFrequncey import cut_samplerate


# trip data from porto
def load_gps_data_porto(filename, crs, to_crs):
    '''
    read trajectory from csv file
    '''
    print('loading Porto GPS Trajectory')
    col_names = ['obj_id', 'lat', 'lon', 'timestamp', 'unknown1', 'unknow2']
    trip = pd.read_csv(filename, header=None, names=col_names)
    trip.drop(['unknown1', 'unknow2'], axis=1, inplace=True)
    trip['geometry'] = trip.apply(lambda z: Point(z.lon, z.lat), axis=1)
    trip = gpd.GeoDataFrame(trip, crs=crs)
    trip.to_crs(to_crs, inplace=True)
    return trip


# calculate the great circle distance between consecutive gps points in the trip
def calculate_great_circle_distance(trip):
    '''
    input: the trip
    output: a list with distances between consecutive points
    '''
    great_circle_distances = [trip.iloc[i]['geometry'].distance(trip.iloc[i+1]['geometry']) for i in range(len(trip)-1)]
    return great_circle_distances


def load_gps_data_seattle(filename, crs, to_crs):
    print('loading Seattle GPS Trajectory')
    gps_track = pd.read_csv(filename,
                            header=None,
                            names=['Data(UTC)', 'Time(UTC)', 'lat', 'lon'],
                            skiprows=[0],
                            delim_whitespace=True)
    # string to datetime
   
    # gps_track=cut_samplerate(gps_track,sample_rate)
    gps_track['datetime'] = gps_track.apply(
        lambda row: datetime.strptime(row['Data(UTC)'] + ' ' + row['Time(UTC)'], '%d-%b-%Y %H:%M:%S'), axis=1)
    gps_track.drop(['Data(UTC)', 'Time(UTC)'], axis=1, inplace=True)
    # datetime to unix time
    gps_track['timestamp'] = gps_track.apply(lambda row: time.mktime(row['datetime'].timetuple()), axis=1)
    gps_track['geometry'] = gps_track.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    gps_track = gpd.GeoDataFrame(gps_track, crs=crs)
    gps_track.to_crs(to_crs, inplace=True)


    return gps_track


def load_gps_data_melbourne(filename, crs, to_crs):
    print('loading Melbourne GPS Trajectory')
    gps_track = pd.read_csv(filename,
                            header=None,
                            names=['timestamp', 'lat', 'lon'],
                            skiprows=[0],
                            delim_whitespace=True)
    gps_track['geometry'] = gps_track.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    gps_track['gps'] = gps_track['geometry']
    gps_track = gpd.GeoDataFrame(gps_track, crs=crs)
    gps_track.to_crs(to_crs, inplace=True)
    return gps_track

