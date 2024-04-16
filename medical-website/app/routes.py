# app/routes.py
import hashlib
import time
from flask import jsonify, redirect, render_template, request, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import requests
from app import app, login_manager, User, db, Message
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename


bcrypt = Bcrypt(app)


load_dotenv()

API_KEY = os.getenv('API_KEY')
TWILIO_API=os.getenv('TWILIO_API')
TWILIO_ID=os.getenv('TWILIO_ID')
TWILIO_PHONE = '+447883319816'


def run_conversation(prompt):
    client = OpenAI(api_key=API_KEY)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    response_text = response.choices[0].message.content
    return response_text


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
@login_required
def index():
    username = current_user.username
    imgpath = current_user.imgpath
    bio = current_user.bio
    return render_template('index.html', username=username, imgpath=imgpath, bio=bio)


@app.route('/home')
@login_required
def home():
    return render_template('home.html')


@app.route('/chat')
@login_required
def chat():
    user = current_user
    chat_history = Message.query.filter_by(sender=user).all()
    for message in chat_history:
        if message.message_type == 'user':
            print("User:", message.content)
        else:
            print("chatgpt:", message.content)
        message.content = message.content.replace('\\n', '<br>')
    if not chat_history:
        message = Message(content="Hello, I will be your biology assistant. Ask me anything to begin!", sender=user, message_type='server')
        db.session.add(message)
        db.session.commit()
        chat_history = [message]
    return render_template('chat.html', chat_history=chat_history)


@app.route('/submit-message', methods=['POST'])
@login_required
def submit_message():
    message_content = request.form['message']

    user = current_user
    message = Message(content=message_content, sender=user, message_type='user')
    db.session.add(message)
    db.session.commit()

    response = run_conversation(prompt=message_content)

    message = Message(content=response, sender=user, message_type='server')
    db.session.add(message)
    db.session.commit()

    return redirect(url_for('chat'))


@app.route('/mcq')
@login_required
def mcq():
    return render_template('mcq.html')


@app.route('/qst', methods=['POST', 'GET'])
@login_required
def qst():
    if request.method == 'POST':
        return redirect(url_for('chat'))
    return render_template('qst.html')


UPLOAD_FOLDER = 'app/static/assets/img/profilepics'
ALLOWED_EXTENSIONS = ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'svg', 'webp']

def allowed_file(filename):
    file_ext = filename.rsplit('.', 1)[1].lower()
    return '.' in filename and file_ext in ALLOWED_EXTENSIONS

def generate_filename(username, extension):
    unique_string = f"{username}{time.time()}"
    hashed_string = hashlib.sha256(unique_string.encode()).hexdigest()
    return f"{hashed_string}.{extension}"

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    imgpath = current_user.imgpath
    if request.method == 'POST':
        username = request.form.get('username').strip()
        bio = request.form.get('bio').strip()
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and allowed_file(file.filename):
                extension = file.filename.rsplit('.', 1)[1].lower()
                filename = generate_filename(username, extension)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                imgpath = f'assets/img/profilepics/{filename}'
                if current_user.imgpath and current_user.imgpath != imgpath:
                    old_img_path = os.path.join(UPLOAD_FOLDER, current_user.imgpath.split('/')[-1])
                    if os.path.exists(old_img_path):
                        os.remove(old_img_path)
        if username!="" and username != current_user.username:
            current_user.username = username
        if bio and bio != current_user.bio:
            current_user.bio = bio
        if imgpath and imgpath != current_user.imgpath:
            current_user.imgpath = imgpath
        db.session.commit()
        return redirect(url_for('profile'))
    return render_template('profile.html', imgpath=imgpath)


@app.route('/future')
@login_required
def future():
    return render_template('future.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check your username and password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/sign-up', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        existing_user_username = User.query.filter_by(username=username).first()
        existing_user_email = User.query.filter_by(email=email).first()

        if existing_user_username:
            flash('Username is already taken. Please choose a different one.', 'danger')
        elif existing_user_email:
            flash('Email is already taken. Please choose a different one.', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password, imgpath="assets/img/avataaars.svg", bio="Enter bio in the Profile Section...")
            db.session.add(user)
            db.session.commit()

            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('sign-up.html')
