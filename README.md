# ML assignment - OCR

## Approach

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/images/dl.JPG" >

- The model ([Segmentation model].https://github.com/qubvel/segmentation_models.pytorch.) with weigths of 'imagenet' as starting point is trained on our dataset as it bears the flexibility to choose various encoders with flexible choice of depth of encoders. Although, conclusively, since the features required are limited in principle to classify : text, bar_code and qr_code, almost any shallow encoder could be just as good. 
- Label masks are constructed using the labels(with images as key and annotations as value) were provided.
- Binary Cross Entropy along with Dice Loss is used as a criterion for minimization.
- The validation Dice Metric that is achieved is ~0.96 
-  After the model is trained to obtain a dice metric of ~0.96 the checkpoint is saved(link below). In inference, the cv2 find contours method is used recursively to obtain the bounding boxes for all the *artefacts* of the test image and whichever channel bears the maximum value(i.e the output- segmentation head of trained model consists of 3 channels dedicated to the 3 classes of text, bar and qr_code each) for that bounding region, that channel is the classification result(text, bar_code or qr_code).  

## Training and Validation:

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/images/metricss.JPG" >

## Arriving at the bounding boxes during prediction phase
Using the function *cv2contour()* in predict.py, bounding boxes for artefacts are obtained using recursive trick as shown below:

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/images/sad.JPG" >

## Input - Output
- Default argument text_on_plot is False for the function *inference*
```
localization,image_array=inference(test_dataloader,text_on_plot=False)
for i in image_array:
    figure(figsize=(20, 44), dpi=50)
    plt.imshow(i)
    plt.show()
``` 
<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/images/ocr.JPG" >

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/images/ocr2.JPG" >

- text_on_plot=True returns
```
localization,image_array=inference(test_dataloader,text_on_plot=True)
for i in image_array:
    figure(figsize=(20, 44), dpi=50)
    plt.imshow(i)
    plt.show()
```
For text recognition purposes, image is pre-processed in two stages:
 - image thresholding
 - image rescaling  

<img src="https://raw.githubusercontent.com/Siddicus/OCR_Classify/master/images/plottrue.JPG" >

## Checkpoint URL

-> "https://github.com/Siddicus/OCR_Classify/releases/download/1/cls_res50.ckpt"

- Predict.py Usage
```
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inference script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('img_path', type=str, help='path to the image') # link to the folder contianing the images
    parser.add_argument('checkpoint_path', type=str, help='path to your model checkpoint') # URL of checkpoint given above
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    localization,image_array= main(args)
    #localization -> ["image_name.jpg":[{"type": "text", "geometry": [.1, .1, .5, .15]}, {"type": "qr_code", "geometry": [.85, .85, .95, .95]}],......]
    # image_array -> numpy array with predictions/plots on the original corresponding image 

```
- Train.py Usage

```
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Model training script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img_folder', type=str, help='path to image folder') # path to the image folder
    parser.add_argument('label_path', type=str, help='path to label file')  # label's(in json format) path 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
```
