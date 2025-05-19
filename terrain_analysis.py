'insert copywrite here'


import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from sklearn.ensemble import RandomForestClassifier
import argparse
import scipy
from scipy.ndimage import distance_transform_edt




def convert_to_rasterio(raster_data, template_raster):
    """
    Reads raster data from a template raster and updates the provided numpy array.

    Parameters:
    raster_data (numpy.ndarray): A numpy array to store the raster data.
    template_raster (rasterio.io.DatasetReader): An open rasterio dataset to read data from.

    Returns:
    numpy.ndarray: The updated numpy array containing the raster data.
    """
    # Read the first band of the template raster
    band_data = template_raster.read(1)
    
    # Update the provided numpy array with the raster data
    np.copyto(raster_data, band_data)
    
    return raster_data

def extract_values_from_raster(raster, shape_object):
    """
    Extracts raster values at the coordinates of the given shapes.

    Parameters:
    raster (rasterio.io.DatasetReader): The raster file to sample values from.
    shape_object (iterable): A collection of geometries (e.g., GeoPandas GeoSeries or GeoDataFrame).

    Returns:
    list: A list of raster values corresponding to the input shapes.
    """
    coordinate_list = []
    # Extract coordinates from each shape
    for shape in shape_object:
        if hasattr(shape, "geometry"):  # If shape is a GeoPandas row
            shape = shape.geometry
        if shape.is_empty:
            continue
        # Get the coordinates of the shape's centroid (or use another method if needed)
        x_coordinate, y_coordinate = shape.centroid.x, shape.centroid.y
        coordinate_list.append((x_coordinate, y_coordinate))

    # Sample raster values at the extracted coordinates
    values = raster.sample(coordinate_list)

    # Convert sampled values to a list
    current_values = [value[0] for value in values]

    return current_values


def make_classifier(x, y, verbose=False):
    # Using random forest classifier
    Random_Forest = RandomForestClassifier(verbose=verbose)
    Random_Forest.fit(x, y)
    return Random_Forest

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
  
    stacked_data = np.stack([topo, geo, lc, dist_fault, slope], axis=-1)
    rows, cols, bands = stacked_data.shape
    reshaped_data = stacked_data.reshape(-1, bands)  # Reshape to (num_pixels, num_features)

    # Predict probabilities using the classifier
    probabilities = classifier.predict_proba(reshaped_data)[:, 1]  

   
    prob_raster = probabilities.reshape(rows, cols)

    return prob_raster
    

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
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

    # Ensure the shape object has a valid shape attribute
    if not hasattr(shape, "shape"):
        raise ValueError("The 'shape' parameter must have a 'shape' attribute.")

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
