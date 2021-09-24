from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

import os

app = FastAPI()
model = ResNet50(weights='imagenet')
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/")
async def root(request: Request):
    
    return templates.TemplateResponse("index.html", {'request': request,})
@app.post("/scorefile/")
async def create_upload_files(request: Request,file: UploadFile = File(...)):
    if 'image' in file.content_type:
        contents = await file.read()
        filename = 'static/' + file.filename
        with open(filename, 'wb') as f:
            f.write(contents)
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    result = decode_predictions(preds, top=3)[0]
    return templates.TemplateResponse("predict.html", {"request": request,"result":result,})

