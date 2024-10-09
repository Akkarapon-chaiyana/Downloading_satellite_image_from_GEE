import google.auth
import ee
from google.api_core import exceptions, retry
from typing import Tuple
import sys
import subprocess
from google.api_core import exceptions
import concurrent.futures
from numpy.lib.recfunctions import structured_to_unstructured
import requests
import numpy as np
from osgeo import gdal
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

ee.Initialize(project = 'tony-1122')

# Define a custom error handler function that does nothing
def handleError(err_class, err_num, err_msg):
    pass


def ee_init() -> None:
        """Authenticate and initialize Earth Engine with the default credentials."""
        # Use the Earth Engine High Volume endpoint.
        #   https://developers.google.com/earth-engine/cloud/highvolume
        credentials, project = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/earthengine",
            ]
        )
        ee.Initialize(
            credentials.with_quota_project(None),
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )


@retry.Retry(deadline=1 * 60)  # seconds
def get_patch(
        image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int
    ) -> np.ndarray:
        """Fetches a patch of pixels from Earth Engine.
        It retries if we get error "429: Too Many Requests".
        Args:
            image: Image to get the patch from.
            lonlat: A (longitude, latitude) pair for the point of interest.
            patch_size: Size in pixels of the surrounding square patch.
            scale: Number of meters per pixel.
        Raises:
            requests.exceptions.RequestException
        Returns: The requested patch of pixels as a NumPy array with shape (width, height, bands).
        """


        point = lonlat
        #print("requesting")
        url = image.getDownloadURL(
            {
                "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
                "dimensions": [patch_size, patch_size],
                "format": "NPY",
            }
        )

        # If we get "429: Too Many Requests" errors, it's safe to retry the request.
        # The Retry library only works with `google.api_core` exceptions.
        response = requests.get(url)
        #print("got response")
        if response.status_code == 429:
            raise exceptions.TooManyRequests(response.text)

        # Still raise any other exceptions to make sure we got valid data.
        response.raise_for_status()
        return np.load(io.BytesIO(response.content), allow_pickle=True), point.buffer(scale * patch_size / 2, 1).bounds(1)

def writeOutput(raster,out_file, patch_size,coords):
    """
    Create a new GeoTIFF file with the given filename and writes the given data into it.

    Args:
        counter (int): A counter to create a unique filename for each output file.
        out_file (str): The output filename to be created for each patch.
        patch_size (int): The size of the patch to be written to the output file.
        overlap_size (int): The size of the overlap between the patches.

    Returns:
        None
    """

    xmin = coords[0][0] #+ (overlap_size * pixel_size)
    xmax = coords[1][0] #- (overlap_size * pixel_size)
    ymin = coords[0][1] #+ (overlap_size * pixel_size)
    ymax = coords[2][1] #- (overlap_size * pixel_size)

    coords = [xmin,ymin,xmax,ymax]

    # Create a new GDAL driver for GeoTIFF
    driver = gdal.GetDriverByName("GTiff")

    l = raster.shape[2] 
    out_raster = driver.Create(out_file, patch_size, patch_size, l, gdal.GDT_Float32)

    # Set the spatial reference system (optional)
    out_raster.SetProjection("EPSG:4326")


    # Set the extent of the file
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / (patch_size ), 0, ymax, 0, -(ymax - ymin) / (patch_size)))

    compress = "LZW"
    options = ["COMPRESS=" + compress]

    layer = raster

    for i in range(0,l,1):
        out_band = out_raster.GetRasterBand(i+1)
        out_band.WriteArray(layer[:,:,i])

    
    out_raster = None
    
    
def writeOutputSingle(raster,out_file, patch_size,coords):
    """
    Create a new GeoTIFF file with the given filename and writes the given data into it.

    Args:
        counter (int): A counter to create a unique filename for each output file.
        out_file (str): The output filename to be created for each patch.
        patch_size (int): The size of the patch to be written to the output file.
        overlap_size (int): The size of the overlap between the patches.

    Returns:
        None
    """

    xmin = coords[0][0] #+ (overlap_size * pixel_size)
    xmax = coords[1][0] #- (overlap_size * pixel_size)
    ymin = coords[0][1] #+ (overlap_size * pixel_size)
    ymax = coords[2][1] #- (overlap_size * pixel_size)

    coords = [xmin,ymin,xmax,ymax]

    # Create a new GDAL driver for GeoTIFF
    driver = gdal.GetDriverByName("GTiff")


    out_raster = driver.Create(out_file, patch_size, patch_size, 1, gdal.GDT_Int16)

    # Set the spatial reference system (optional)
    out_raster.SetProjection("EPSG:4326")

    # Set the extent of the file
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / (patch_size ), 0, ymax, 0, -(ymax - ymin) / (patch_size)))

    compress = "LZW"
    options = ["COMPRESS=" + compress]

    layer = raster


    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(layer)

    
    out_raster = None    
    
    
patch_size = 512


def addMyClass(feat):
    year = 2023
    month = 1
    d = ee.Date.fromYMD(year,month,1)
    return feat.set("class",1).set("system:time_start",d).set("year",year).set("month",month)


def getData(i): 
    mySamples = ee.FeatureCollection("projects/ee-akkaraponchaiyana/assets/Tony/AlertMonthly_20210200").randomColumn("random").sort("random").map(addMyClass)
    samples = mySamples.toList(5000)
    sample  = ee.Feature(samples.get(i))
        
    img_path_before = f"/content/drive/MyDrive/Clay_mangrove/Images/s2/"+str(i).zfill(4) + ".tif"
    img_path_label = f"/content/drive/MyDrive/Clay_mangrove/Images/label/"+str(i).zfill(4) + ".tif"

    geometry = sample.geometry()
    
    lonlat = geometry

    CLEAR_THRESHOLD = 0.80
    QA_BAND = 'cs_cdf';
        
    beforeStartDate = ee.Date.fromYMD(2023,10,1)
    beforeEndDate = ee.Date.fromYMD(2023,12,31)

    def maskData(img):
       return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));


    def s2MaskData(img):
      scl = img.select('SCL')  # Select the Scene Classification Layer
      # Keep only Vegetation (4), Bare Soils (5), and Water (6) pixels
      keep_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
      return img.updateMask(keep_mask)

    s2 =  ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geometry).filterDate(beforeStartDate,beforeEndDate).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",40)).sort("CLOUDY_PIXEL_PERCENTAGE")
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').filterBounds(geometry).filterDate(beforeStartDate,beforeEndDate)
    #print(s2.size().getInfo())
    s2 = s2.linkCollection(csPlus, [QA_BAND]).map(maskData)
    #s2 = ee.ImageCollection(s2.map(s2MaskData))
    #print(s2.size().getInfo())  
    
    BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    
    s2Before = s2.filterDate(beforeStartDate,beforeEndDate)
    
    s2Before = s2Before.median().select(BANDS)
        
    patch, bounds = get_patch(s2Before, lonlat, patch_size, 10)
    imagePatch = structured_to_unstructured(patch) 
    coords = np.array(bounds.getInfo().get("coordinates"))[0]
    writeOutput(imagePatch,img_path_before, patch_size,coords)
    beforeImg = imagePatch
    print(f"s2 images: {i}",beforeImg.shape)
 
    alerts = ee.Image("projects/servir-mekong/UMD/Loss_C02/2021")
    
    patch, bounds = get_patch(alerts, lonlat, patch_size, 10)
    imagePatch = structured_to_unstructured(patch).squeeze()
    coords = np.array(bounds.getInfo().get("coordinates"))[0]
    writeOutputSingle(imagePatch,img_path_label, patch_size,coords)
    label_target = imagePatch
    print(f"label: {i}",label_target.shape)
    

# Adjust the number of threads as per your system's capabilities
NUM_THREADS = 8

#getData(38)  


def process_data(i):
    try:
        getData(i)
    except Exception as e:
        print(f"Error processing index {i}: {e}")

def main():
    start_index = 0
    end_index = 100

    # Using ThreadPoolExecutor to run `getData` function in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_data, i) for i in range(start_index, end_index)]
        
        # Wait for the tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # This raises any exception thrown in `process_data`
            except Exception as e:
                print(f"Exception in thread: {e}")

if __name__ == '__main__':
    main()


