from flask import Flask, render_template

app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024