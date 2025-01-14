from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.presets import get_automm_presets
from IPython.display import Image, display
from ray import tune
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import yaml
import uuid
import zipfile
import os
import datetime
import torch
import onnx
import onnxruntime

def main():
    predictor = MultiModalPredictor.load("e:/Current_Workdir/palm-fruit-classification/model/train_2024-10-08_18-03-14")
    image_path = 'e:\\Current_Workdir\\palm-fruit-classification\\data\\intermediate\\valid\\empty_bunch\\IMG_20220803_112710_crop_0_jpg.rf.bfef2ca25d24fefe9a8c64c68c5bb66f.jpg'
    predictions = predictor.predict({'image': [image_path]})
    print(predictions)
    
if __name__ == '__main__':
    main()