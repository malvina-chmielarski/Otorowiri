import ee
import geemap

# Initialize
ee.Authenticate()
ee.Initialize()

# Define region
region = ee.Geometry.Rectangle([115.294, -29.15, 115.858, -29.95])

# Load AET dataset (e.g., SSEBop)
aet = ee.ImageCollection("projects/ee-earthengine-datasets/assets/WRI/GFW/aet_monthly") \
          .filterDate('1987-05-01', '1987-09-31') \
          .select('aet') \
          .mean()

# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=aet.clip(region),
    description='Mean_AET_1987_Wet',
    folder='GEE_exports',
    scale=1000,
    region=region.getInfo()['coordinates'],
    fileFormat='GeoTIFF'
)
task.start()