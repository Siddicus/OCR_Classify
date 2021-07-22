# ML assignment - OCR

## Approach

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/dl.JPG" >

- The model ([Segmentation model].https://github.com/qubvel/segmentation_models.pytorch.) with weigths of 'imagenet' as starting point is trained on our dataset as it bears the flexibility to choose various encoders with flexible choice of depth of encoders. Although, conclusively, since the features required are limited in principle to classify : text, bar_code and qr_code, almost any shallow encoder could be just as good. 
- Label masks are constructed using the labels(with images as key and annotations as value) were provided.
- Binary Cross Entropy along with Dice Loss is used as a criterion for minimization.
- The validation Dice Metric that is achieved is ~0.96 
- Now, the cv2 find contours method is used recursively to obtain the bounding boxes for all the *artefacts* and whichever channel bears the maximum value ( with model.eval() and then acting model on the input image))  for that bounding region, that channel is the classification result(text, bar_code or qr_code).  

## Training and Validation:

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/metricss.JPG" >

## Arriving at the bounding boxes during prediction phase
Using the function *cv2contour()* in predict.py, bounding boxes for artefacts are obtained using recursive trick as shown below:

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/sad.JPG" >

## Input - Output

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/ocr.JPG" >

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/ocr2.JPG" >
