import glob
import json
import random
import cv2
import pytesseract
import os
languages = ["Assamese","Bangla","Gujarati","Gurumukhi","Hindi","Kannada","Malayalam","Manipuri","Marathi","Oriya","Tamil","Telugu","Urdu"]

for language in languages:
    # os.system(f"mkdir {language}")
    dirs = random.sample(glob.glob("/scratch/Consortium_dataset/" + language + "/*/*_*/*.tif"),10)
    dic1 = {}
    for c,dir in enumerate(dirs):
        image = cv2.imread(dir)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = []
        id = language + str(c)
        dic1[id] = {}
        dic1[id]["imagePath"] = "/home/ajoy" + dir[8:]
        dic1[id]["regions"] = []
        results = pytesseract.image_to_data(dir, output_type=pytesseract.Output.DICT)
        for i in range(0, len(results["text"])):
            # extract the bounding box coordinates of the text region from
            # the current result
            dic_sub = {}
            dic_sub["outputs"] = {}
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            dic_sub["groundTruth"] = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
            dic_sub["id"] = id
            dic1[id]["regions"].append(dic_sub)
            dic_sub["regionLabel"] = "region"
            conf = int(results["conf"][i])
            if conf >0:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # dic1[language].append(bboxes)
        cv2.imwrite(language + "/" + str(c) + ".tif", image)
        # print(id)
    outfile = open(language + "_tesseract.json","w")
    json.dump(dic1, outfile, indent = 6)
    outfile.close()

  
outfile.close()