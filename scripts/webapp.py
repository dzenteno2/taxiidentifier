'''
Created on Mar 7, 2015

@author: Daniel
'''

import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug import secure_filename
from supervised_learning.Supervised import Supervised
from base.dataset import Dataset
from base import preprocessing

UPLOAD_FOLDER = '/Users/Daniel/Documents/HackCDMX/uploads'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                f_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(f_name)
                
                # Once is saved, process
                
                supervised = Supervised()
                supervised.load()
                dataset = Dataset()
                letters = preprocessing.search_numbers(f_name,
                                             supervised,
                                             dataset)
                
                f = {'placa': letters}
                
                return jsonify(**f)
                            
        except Exception as e:
            print e.trace()
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
            
if __name__ == '__main__':
    app.run()