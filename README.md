# VISORD
Project for the computer vision third-year course

## Execution
Simply type :  
```
python text_detection.py
```  
The script will process all the images from the images directory and place the result in an new output folder.

## Technical stack
The script was developped under :  
- Python 2.7  
- CV2 3.4.0

The rest of the modules are in the requirements.txt file.

## Remarks:
By default the code will plot figures at each step. Please set VERBOSE = 0 (line 30) at the beginning of the code to prevent from plotting.

The code handles only png and jpg files for now.
