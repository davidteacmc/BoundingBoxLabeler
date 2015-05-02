bounding-box-labeler
======================
Manual labeler of object bounding boxes in images

### Updates

05/01/2015 (v0.1) : generate text output in addition to XML  
07/18/2014 (v0.0) : first version

### Steps
1. List all image paths in a file named "list.txt". One file for each line. An example of list.txt:
>`D:\images\00000002.jpg`  
>`D:\images\00000004.jpg`  
>`D:\images\00000006.jpg`  
>`D:\images\00000008.jpg`  
>`D:\images\00001014.jpg`  
>`..\frames\00001155.jpg`  
>`..\frames\00001157.jpg`  
>`..\frames\00001162.jpg`  
>`..\frames\00001421.jpg`  
>`..\frames\00001470.jpg`  

2. Keep the DLL files in the same directory as BoundingBox.exe.

3. Open BoundingBox.exe.

4. Type the image to start. It needs to be exactly the same as one of the paths in list.txt, for example, D:\images\00000002.jpg.

5. Use the mouse to label bounding boxes in the image window.  
1) Click to locate the top-left corner.  
2) Click again to locate the bottom-right corner. The coordinates (x1, y1, x2, y2) will be printed.  
3) Repeat 1) and 2) if there are multiple objects.  

* Spacebar: proceed to the next image
* Right-click: undo
* ESC: quit

6. Results are saved in a text file named "bounding_boxes.txt"

