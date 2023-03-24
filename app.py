from tkinter import Button
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash import no_update
import base64
import random
import io
from io import BytesIO
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization, Activation
from keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.models import load_model

classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3]]) 
def names(number):
    if(number == 0):
        return 'a Glioma tumor'
    elif(number == 1):
        return 'a Meningioma tumor'
    elif(number == 2):
        return 'no tumor'
    elif(number == 3):
        return 'a Pituitary tumor'  
        

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
white_button_style = {'background-color': 'Tomato',
                      'color': 'black',
                      'height': '40px',
                      'width': '150px',
                      'margin-top': '50px',
                      'margin-left': '50px'}

app.layout = html.Div([
    
    html.H1(children='BRAIN TUMOR CLASSIFIER', style={'textAlign': 'center'
        }),
    
    dcc.Markdown('''
                ### Step 1: Import a single image using upload button
                ### Step 2: Wait for prediction and Remedies
    '''),
    

    dcc.Upload(
        id='upload-image',
        children=html.Div([
             html.Button('Upload here', id='submit-val', n_clicks=0,style= white_button_style),
        ]),
        style={
            
            'width': '95%',
            'height': '110px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '15px',
            'textAlign': 'center',
            'margin': '20px'
        }, 
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload', style={'position':'absolute', 'left':'200px', 'top':'250px'}),
    
    html.Div(id='prediction', style={'position':'absolute', 'left':'800px', 'top':'310px', 'font-size':'x-large'}),
    html.Div(id='prediction2', style={'position':'absolute', 'left':'800px', 'top':'365px', 'font-size': 'x-large'}),
    
    html.Div(id='facts', style={'position':'absolute', 'left':'800px', 'top':'465px', 'font-size': 'large',\
                               'height': '200px', 'width': '500px'}),

])

def parse_contents(contents):
    
    return html.Img(src=contents, style={'height':'450px', 'width':'450px'})


@app.callback([Output('output-image-upload', 'children'), Output('prediction', 'children'), Output('prediction2', 'children'), 
              Output('facts', 'children')],
              [Input('upload-image', 'contents')])

def update_output(list_of_contents):        
    
    if list_of_contents is not None:
        children = parse_contents(list_of_contents[0]) 
         
        img_data = list_of_contents[0]
        img_data = re.sub('data:image/jpeg;base64,', '', img_data)
        img_data = base64.b64decode(img_data)  
        
        stream = io.BytesIO(img_data)
        img_pil = Image.open(stream)
        
        
        #Load model, change image to array and predict
        model = load_model('model_final.h5') 
        dim = (150, 150)
        
        img = np.array(img_pil.resize(dim))
        
        x = img.reshape(1,150,150,3)
       

        answ = model.predict(x)
        classification = np.where(answ == np.amax(answ))[1][0]
         
        
        #Second prediction and facts about tumor if there is
        if classification==0:
            pred=str(round(answ[0][classification]*100 ,3)) + '% confidence there is ' + names(classification)+ '    '+ 'Stage:2'
            facts = 'occurrence  -  brain and spinal cord.\
                     Type -  malignant (cancerous)\
                     Treatment - Chemotherapy  in combination with radiation therapy .\
                     Estimated cost - 10-12 lakhs'
            
            no_tumor = str(round(answ[0][2]*100 ,3))
            pred2 = no_tumor + '% confidence there is no tumor'
            
        elif classification==1:
            pred=str(round(answ[0][classification]*100 ,3)) + '% confidence there is ' + names(classification) + '    '+ 'Stage:3'
            facts = 'Meningioma occurrence -  arises from the meninges\
                     Type -   Benign (non-cancerous)\
                     Treatment - commonly treated with surgery,Radiation therapy is also used\
                     Estimated cost -  2.5-5 lakhs'

            no_tumor = str(round(answ[0][2]*100 ,3))
            pred2 = no_tumor + '% confidence there is no tumor'
        
        elif classification==3:
            pred=str(round(answ[0][classification]*100 ,3)) + '% confidence there is ' + names(classification)
            facts = 'Pituitary tumors are abnormal growths that develop in your pituitary gland.\
                    Most pituitary tumors are noncancerous (benign) growths that remain in your pituitary\
                    gland or surrounding tissues.'
            no_tumor = str(round(answ[0][2]*100 ,3))
            pred2 = no_tumor + '% confidence there is no tumor'
        
        else:
            pred=str(round(answ[0][classification]*100 ,3)) + '% confidence there is ' + names(classification)
            facts=None
            pred2 = None
        
        return children, pred, pred2, facts
    
    else:
        return (no_update, no_update, no_update, no_update)  

if __name__ == '__main__':
    app.run_server(debug=True)
