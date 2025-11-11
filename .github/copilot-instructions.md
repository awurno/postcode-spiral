Create a tool to generate a map conaining a line connecting postcodes within a specific zone. 

The postal code data from the netherlands can be taken from the ogc api: https://api.pdok.nl/cbs/postcode4/ogc/v1

or the WFS: https://service.pdok.nl/cbs/postcode4/2024/wfs/v1_0?request=GetCapabilities&service=WFS

the data from 2024 is fine. 

The tool should be run from the commandline with argument either a province name, municipality name or a bbox (xmin, xmax, ymin, ymax). 
