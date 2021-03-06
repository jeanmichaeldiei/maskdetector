import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import web_predict
import cv2
import time
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = web_predict.create_model()
video_stream = web_predict.MyVideoStream(model)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	video_stream.do_preds = False
	time.sleep(0.1)
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(path)
		#Doing inference on the image
		img, flag = web_predict.predict_image(model,path)
		# Saving the image 
		cv2.imwrite(path, img) 
		if flag:
			flash('Model successfully displayed predictions!')
		else:
			flash('No strong predictions could be displayed.')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

# #Webcam Stuffs
def gen(camera):
	while True:
		frame = camera.predict_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	return Response(gen(video_stream),
			mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_controller',methods=['POST'])
def webcam_controller():
	json = request.get_json()
	print(json)
	if json and json['status'] == "true":		
		video_stream.do_preds = True
	else:
		video_stream.do_preds = False
	return (''), 200
if __name__ == "__main__":
    app.run()