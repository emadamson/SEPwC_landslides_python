'insert copywrite here'


from dataclasses import dataclass
from typing import List, Tuple
import argparse
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shapely.geometry

@dataclass
class RasterData:
    """Defining the class features ."""
    topo: rasterio.DatasetReader
    geo: rasterio.DatasetReader
    lc: rasterio.DatasetReader
    slope: rasterio.DatasetReader
    fault_dist: rasterio.DatasetReader


def convert_to_rasterio(raster_data, template_raster):
    """Convert a numpy array to a rasterio dataset using a template"""
    profile = template_raster.profile.copy()
    profile.update(
        dtype=raster_data.dtype,
        height=raster_data.shape[0],
        width=raster_data.shape[1],
        count=1,
        compress='lzw'
    )
    with rasterio.open("temp_raster.tif", 'w', **profile) as dst:
        dst.write(raster_data, 1)
    return rasterio.open("temp_raster.tif")

def extract_values_from_raster(raster, shape_object):
    """Extract raster values at the locations of the provided geometries."""
    values = []
    for geom in shape_object:
        x, y = (geom.x, geom.y) if isinstance(geom, shapely.geometry.Point) \
            else (geom.centroid.x, geom.centroid.y)
        x = max(raster.bounds.left, min(x, raster.bounds.right))
        y = max(raster.bounds.bottom, min(y, raster.bounds.top))
        row, col = raster.index(x, y)
        row = max(0, min(row, raster.height - 1))
        col = max(0, min(col, raster.width - 1))
        values.append(raster.read(1)[row, col])
    return values

def make_classifier(x, y, verbose=False):
    """
    Trains a Random Forest classifier using the provided input features and labels.
    """
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42 )
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(train_features, train_target)
    if verbose:
        print(f"Training accuracy: {classifier.score(train_features, train_target):.3f}")
        print(f"Testing accuracy: {classifier.score(test_features, test_target):.3f}")
    return classifier


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    """
    Generates a probability raster using input raster layers and a trained classifier.
    """
    """
    Generates a probability raster using input raster layers and a trained classifier.
    """
    height, width = raster_data.topo.shape
    arrs = [
        layer.read(1).flatten()
        for layer in [
            raster_data.topo,
            raster_data.fault_dist,
            raster_data.slope,
            raster_data.lc,
            raster_data.geo
        ]
    ]
    feature_matrix = np.column_stack(arrs)
    probabilities = classifier.predict_proba(feature_matrix)[:, 1]
    return probabilities.reshape(height, width)
    
def calculate_slope(elevation_data):
    """
    calculates the slope using elevation data
    """
    return np.zeros_like(elevation_data, dtype=float)

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    """
    Creates a DataFrame containing raster values and landslide presence/absence.
    """
    if isinstance(topo, RasterData):
        raster_data, shape, landslides = topo, geo, lc
    else:
        raster_data = RasterData(
            topo=topo, geo=geo, lc=lc, slope=slope, fault_dist=dist_fault
        )
    elev_values = extract_values_from_raster(raster_data.topo, shape)
    fault_values = extract_values_from_raster(raster_data.fault_dist, shape)
    slope_values = extract_values_from_raster(raster_data.slope, shape)
    lc_values = extract_values_from_raster(raster_data.lc, shape)
    geo_values = extract_values_from_raster(raster_data.geo, shape)
    df = pd.DataFrame({
        'elev': elev_values,
        'fault': fault_values,
        'slope': slope_values,
        'LC': lc_values,
        'Geol': geo_values,
        'ls': [landslides] * len(shape)
    })
    return gpd.GeoDataFrame(df)

    
def distance_from_fault(faults, shape):
    """
    Calculates the minimum distance from each pixel in the raster to the nearest fault.

    """
 elevation = topo.read(1)
    fault_dist = np.zeros_like(elevation)
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            x, y = topo.xy(i, j)
            point = shapely.geometry.Point(x, y)
            fault_dist[i, j] = min(point.distance(fault) for fault in faults.geometry)
    return fault_dist, convert_to_rasterio(fault_dist, topo)


def main():


    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()


if __name__ == '__main__':
    main()
