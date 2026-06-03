why am i making you do this?
basically, 2018 weinstein data from milliontree (weinstein unsupervised) is lidar data from ALASKA and Washington DC. My issue with that is, these are not fire prone regions, thus the 
landscape isn't going to have any dead trees, burn marks, etc. that are definitely going to 
preset in our work. so we pivot (don't worry, this file is the end to end workflow for getting data, i have it on my machine already). to what?

Weinstein 2020--deepforest prediction release on all NEON sites (prediction csv are published on zenodo, see below). I select TEAK (Washington), SOAP (california, neay yosemite), and YELL (yellowstone) as the sites. I hand inspected them to make sure they had dead trees. 

Another benefit (more like a good side effect) of pulling data this way. we can pull the hyperspectral data cube, average the infra bands to make a pseudo-nir band. that way, we can actually make 4 band images that will be similar to NAIP (we can tweak the resolutions to make it 0.6m gsd too) and pretrain on that! i think this makes sense.


first, go to https://zenodo.org/records/3765872#.X2J1zZNKjOQ and download
- TEAK.csv
- SOAP.csv
- YELL.csv

about 1.5gigs. this is only the labels.

now, download the neon tiles. 
use modal_neon_dl.py. change site to "TEAK", then "SOAP", then "YELL". each will  finishes in like 2 minutes, >100gb download each btw. 

now, we have to clean the dataset. 

1. filter out bboxes that are on dead trees (NDVI filtering trick). you can look at label_matching.ipynb to see how that works. the NDVI does indeed filter out dead trees quite effectively. i found that the ndvi threshold was best at 0.6
2. tile up into 256x256 patches, with corresponding labels. original image is 10000x10000 at GSD = 0.1m, so 1kmx1km. we want 256*0.6=153mx153m tiles. we get about 36 patches per tile. 
3. to match the IR to the RGB, we upsample the IR from 1m/pixel to 0.6m/pixel. This isn't that crazy of an upsampling so hopefully its fine.
4. in labels.csv, (where all the bboxes will live), we include lat/lon and time, since clay wants these too. 

those 3 steps will be handled by modal_pipeline.py. once the dl is done. run it. this one will take like 10 minutes.

will create in your volume
patches
|-----TEAK
        |-----img patches
|-----SOAP
        |-----img patches
|-----YELL
        |-----img patches 
labels_SOAP.csv
labels_TEAK.csv
labels_YELL.csv
labels.csv

your data is ready to be used. 



updates to the clay model md. 

1. input to clay will be 256x256. your claude keeps saying 224x224, and yes, that'll work,
but clay is input size agnostic. it can handle 256x256, which is our native resolution.
since we can work in native res, we should do that. 

2. the pre train of the neck + head will be done with 4 band neon images from YELL, SOAP, and TEAK now. if you followed instructions, you should have all that data in modal volume now. yay.

3. also please look into the "neck" stuff. we might not need to upsample. something about stride.