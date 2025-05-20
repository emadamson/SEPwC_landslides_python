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

    Returns:
    numpy.ndarray: A NumPy array representing the probability raster.
    """
    # Stack all input raster layers into a single 2D array (features for the classifier)
    stacked_data = np.stack([topo, geo, lc, dist_fault, slope], axis=-1)
    rows, cols, bands = stacked_data.shape
    reshaped_data = stacked_data.reshape(-1, bands)  # Reshape to (num_pixels, num_features)

    # Predict probabilities using the classifier
    probabilities = classifier.predict_proba(reshaped_data)[:, 1]  # Get probabilities for the positive class

    # Reshape the probabilities back to the original raster shape
    prob_raster = probabilities.reshape(rows, cols)

    return prob_raster
    

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    """
    Creates a DataFrame containing raster values and landslide presence/absence.
    Returns: pandas.DataFrame: A DataFrame containing the raster values as features and a landslide column indicating the presence (1) or absence (0) of landslides
    """
    # Flatten raster data into 1D arrays
    topo_flat = topo.flatten()
    geo_flat = geo.flatten()
    lc_flat = lc.flatten()
    dist_fault_flat = dist_fault.flatten()
    slope_flat = slope.flatten()

    # Create a DataFrame with raster values
    data = pd.DataFrame({
        "topography": topo_flat,
        "geology": geo_flat,
        "landcover": lc_flat,
        "distance_to_fault": dist_fault_flat,
        "slope": slope_flat
    })

    # Extract landslide presence/absence
    raster_shape = topo.shape
    landslide_raster = features.rasterize(
        [(geom, 1) for geom in landslides.geometry],
        out_shape=raster_shape,
        transform=shape.transform,
        fill=0,
        dtype="int32"
    )
    data["landslide"] = landslide_raster.flatten()

    return data
    
def distance_from_fault(faults, shape):
    """
    Calculates the minimum distance from each pixel in the raster to the nearest fault.


    Returns:
    numpy.ndarray: A NumPy array representing the distance raster.
    """
    # Create a binary mask for fault locations
    raster_shape = shape.shape
    fault_mask = features.rasterize(
        [(geom, 1) for geom in faults.geometry],
        out_shape=raster_shape,
        transform=shape.transform,
        fill=0,
        dtype="int32"
    )

    # Calculate the Euclidean distance transform
    distance_raster = distance_transform_edt(fault_mask == 0)

    return distance_raster


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
