import argparse
import ee
import google.auth
from google.api_core import retry
import requests
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

# Initialize Earth Engine with the default project and authentication
ee.Initialize(project = 'tony-1122')

patch_size = 512
NUM_THREADS = 8  # You can adjust this based on your system capabilities

# Retry mechanism for rate-limited requests (HTTP 429)
@retry.Retry(deadline=1 * 60)  # Retry for up to 1 minute
def get_patch(image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int) -> np.ndarray:
    point = lonlat
    url = image.getDownloadURL({
        "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
        "dimensions": [patch_size, patch_size],
        "format": "NPY",
    })
    response = requests.get(url)
    if response.status_code == 429:
        raise requests.exceptions.RequestException("Too many requests, rate limited.")
    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True), point.buffer(scale * patch_size / 2, 1).bounds(1)

def writeOutput(raster, out_file, patch_size, coords):
    xmin, xmax, ymin, ymax = coords[0][0], coords[1][0], coords[0][1], coords[2][1]
    coords = [xmin, ymin, xmax, ymax]
    
    driver = gdal.GetDriverByName("GTiff")
    l = raster.shape[2]
    out_raster = driver.Create(out_file, patch_size, patch_size, l, gdal.GDT_Float32)
    out_raster.SetProjection("EPSG:4326")
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / patch_size, 0, ymax, 0, -(ymax - ymin) / patch_size))
    
    for i in range(0, l):
        out_band = out_raster.GetRasterBand(i+1)
        out_band.WriteArray(raster[:, :, i])

    out_raster = None

def writeOutputSingle(raster, out_file, patch_size, coords):
    xmin, xmax, ymin, ymax = coords[0][0], coords[1][0], coords[0][1], coords[2][1]
    coords = [xmin, ymin, xmax, ymax]
    
    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(out_file, patch_size, patch_size, 1, gdal.GDT_Int16)
    out_raster.SetProjection("EPSG:4326")
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / patch_size, 0, ymax, 0, -(ymax - ymin) / patch_size))
    
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(raster)
    out_raster = None

def addMyClass(feat):
    year = 2023
    month = 1
    d = ee.Date.fromYMD(year, month, 1)
    return feat.set("class", 1).set("system:time_start", d).set("year", year).set("month", month)

def getData(i, location): 
    mySamples = ee.FeatureCollection(location).randomColumn("random").sort("random").map(addMyClass)
    samples = mySamples.toList(mySamples.size())  # Adjusting to handle all samples
    sample = ee.Feature(samples.get(i))
        
    img_path_before = f"/content/Clay_mangrove/Images/s2/{str(i).zfill(4)}.tif"
    img_path_label = f"/content/Clay_mangrove/Images/label/{str(i).zfill(4)}.tif"

    geometry = sample.geometry()
    lonlat = geometry

    CLEAR_THRESHOLD = 0.80
    QA_BAND = 'cs_cdf'
        
    beforeStartDate = ee.Date.fromYMD(2023, 1, 1)
    beforeEndDate = ee.Date.fromYMD(2023, 12, 31)

    def maskData(img):
        return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))

    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geometry).filterDate(beforeStartDate, beforeEndDate).sort("CLOUDY_PIXEL_PERCENTAGE")
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').filterBounds(geometry).filterDate(beforeStartDate, beforeEndDate)
    s2collection = s2.linkCollection(csPlus, [QA_BAND]).map(maskData)
    
    BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    s2Before = s2collection.select(BANDS).median()
        
    patch, bounds = get_patch(s2Before, lonlat, patch_size, 10)
    imagePatch = structured_to_unstructured(patch) 
    coords = np.array(bounds.getInfo().get("coordinates"))[0]
    writeOutput(imagePatch, img_path_before, patch_size, coords)
    print(f"s2 images: {i}", imagePatch.shape)

    mangroves = ee.Image("projects/tony-1122/assets/mangrove_th_project/mangrove_raster_sea")
    patch, bounds = get_patch(mangroves, lonlat, patch_size, 10)
    imagePatch = structured_to_unstructured(patch).squeeze()
    coords = np.array(bounds.getInfo().get("coordinates"))[0]
    # writeOutputSingle(imagePatch, img_path_label, patch_size, coords)
    # print(f"label: {i}", imagePatch.shape)

def process_data(i, location):
    try:
        getData(i, location)
    except Exception as e:
        print(f"Error processing index {i}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process satellite data for a specified location.")
    parser.add_argument("location", type=str, help="Earth Engine FeatureCollection location path")
    args = parser.parse_args()

    location = args.location  # Location provided as an argument
    
    # Get the total number of features in the collection
    mySamples = ee.FeatureCollection(location)
    total_samples = mySamples.size().getInfo()  # Get total number of samples

    print(f"Processing {total_samples} samples from the collection: {location}")
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_data, i, location) for i in range(total_samples)]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception in thread: {e}")

if __name__ == '__main__':
    main()
