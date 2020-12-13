import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import web_predict
import cv2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = web_predict.create_model()
video_stream = web_predict.MyVideoStream(model)
#video_stream = web_predict.VideoCamera()
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
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
		img, flag = web_predict.predict_image(model,path)
		# Saving the image 
		cv2.imwrite(path, img) 
		#print('upload_image filename: ' + filename)
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
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

#Webcam Stuffs
# def gen(camera):
# 	camera.power_on()
# 	while True:
# 		frame = camera.predict_frame()
# 		yield (b'--frame\r\n'
# 				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
# 	return Response(gen(vs),
# 				mimetype='multipart/x-mixed-replace; boundary=frame')
def gen(camera):
	camera.power_on()
	while True:
		frame = camera.predict_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	return Response(gen(video_stream),
				mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()