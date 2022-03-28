import os

# Let's pick the desired backend
# os.environ['USE_TF'] = '1'
os.environ['USE_TORCH'] = '1'
import numpy
import matplotlib.pyplot as plt
import random
import glob
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
languages = ["Assamese","Bangla","Gujarati","Gurumukhi","Hindi","Kannada","Malayalam","Manipuri","Marathi","Oriya","Tamil","Telugu","Urdu"]

predictor = ocr_predictor(pretrained=True)
for language in languages:
    os.system(f"mkdir {language}")
    dirs = random.sample(glob.glob("/scratch/Consortium_dataset/" + language + "/*/*_*/*.tif"),10)
    print(language)
    for c,dir in enumerate(dirs):
        image = cv2.imread(dir)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        doc = DocumentFile.from_images(dirs)
        result = predictor(doc)
    # result.show(doc)
        dic = result.export()
    # outfile = open(language + "/out" + ".json","w")
    # json.dump(dic, outfile, indent = 6)
    # outfile.close()
        page_words = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
        page_dims = [page['dimensions'] for page in dic['pages']]
    # # Get the coords in [xmin, ymin, xmax, ymax]
        words_abs_coords = [
            [[int(round(word['geometry'][0][0] * dims[1])), int(round(word['geometry'][0][1] * dims[0])), int(round(word['geometry'][1][0] * dims[1])), int(round(word['geometry'][1][1] * dims[0]))] for word in words]
            for words, dims in zip(page_words, page_dims)

    ]
    # print(len(words_abs_coords[0]))
        for w in words_abs_coords[0]:
            # print(w[0])
            cv2.rectangle(image,(w[0], w[1]),(w[2], w[3]),(0,255,0))
        cv2.imwrite(language + "/" + str(c) + ".tif", image)