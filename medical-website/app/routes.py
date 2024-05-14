# app/routes.py
import hashlib
import time
from flask import jsonify, redirect, render_template, request, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import requests
from app import app, login_manager, User, db, Message, Doctor, MedicalProfile
from datetime import datetime
from openai import OpenAI
import os
from werkzeug.utils import secure_filename
from sqlalchemy import case


bcrypt = Bcrypt(app)



API_KEY = os.getenv('API_KEY')

TWILIO_API=os.getenv('TWILIO_API')
TWILIO_ID=os.getenv('TWILIO_ID')
TWILIO_PHONE = '+447883319816'

from collections import Counter
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import traceback

# Initialize empty models dictionary
models = {}

# Define model names
model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Support Vector Machine', 'Gaussian Naive Bayes']

# Load models with error handling
for model_name in model_names:
    try:
        model_path = f"models/{model_name}.joblib"
        print(f"Loading {model_name} from {model_path}")
        models[model_name] = load(model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        traceback.print_exc()

all_symptoms = pd.read_csv('data/Training.csv').columns[:-2].tolist()

label_encoder = load('models/label_encoder.joblib')

def predict_symptoms(symptoms_vector):
    scaler = load('models/scaler.joblib')
    symptoms_vector_scaled = scaler.transform(symptoms_vector.reshape(1, -1))
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(symptoms_vector_scaled)
        predictions[model_name] = label_encoder.inverse_transform(prediction)
    return predictions

def encode_symptoms(symptoms):
    encoded_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    return encoded_vector

def predict_symptoms_with_probabilities_and_models(symptoms_vector):
    predictions = predict_symptoms(symptoms_vector)
    prediction_counts = Counter()
    for model_name, prediction in predictions.items():
        prediction_counts[prediction[0]] += 1
    total_predictions = sum(prediction_counts.values())
    probabilities = {prediction: count / total_predictions * 100 for prediction, count in prediction_counts.items()}
    return probabilities, predictions


def test(selected_symptoms = ['knee_pain', 'history_of_alcohol_consumption', 'dehydration', 'vomiting', 'movement_stiffness', 'diarrhoea']):
    # selected_symptoms = ['knee_pain', 'history_of_alcohol_consumption', 'dehydration', 'vomiting', 'movement_stiffness', 'diarrhoea']
    print("\nSelected symptoms:", selected_symptoms)
    encoded_vector = np.array(encode_symptoms(selected_symptoms))
    print("Encoded vector:", encoded_vector)
    probabilities, predictions = predict_symptoms_with_probabilities_and_models(encoded_vector)
    print("Predictions by each model:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction[0]}")
    print("Percentage chances of each unique prediction:")
    returnings = [f"Selected Symptoms: {selected_symptoms}"]
    for prediction, percentage in probabilities.items():
        returnings.append(f"{prediction}: {percentage:.2f}%")
        print(f"Prediction: {prediction}, Percentage: {percentage:.2f}%")
    return returnings
# test()


def run_conversation(prompt):
    client = OpenAI(api_key=API_KEY)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    response_text = response.choices[0].message.content
    return response_text

@app.route('/')
@login_required
def index():
    username = current_user.username
    imgpath = current_user.imgpath
    bio = current_user.bio
    return render_template('index.html', username=username, imgpath=imgpath, bio=bio)


@app.route('/home', methods=['POST', 'GET'])
@login_required
def home():
    user = current_user
    medical_profile = user.medical_profile

    if request.method == 'POST':
        dob_str = request.form['dob']
        dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
        sex = request.form['sex']
        allergies = request.form['allergies']
        medications = request.form['medications']
        emergency_contact_name = request.form['emergency_contact_name']
        emergency_contact_phone = request.form['emergency_contact_phone']
        blood_type = request.form['blood_type']
        smoking_status = request.form['smoking_status']
        if smoking_status == 'false':
            smoking_status = False
        else:
            smoking_status = True

        if not medical_profile:
            medical_profile = MedicalProfile(
                user=user,
                dob=dob,
                sex=sex,
                allergies=allergies,
                medications=medications,
                emergency_contact_name=emergency_contact_name,
                emergency_contact_phone=emergency_contact_phone,
                blood_type=blood_type,
                smoking_status=smoking_status
            )
        else:
            medical_profile.dob = dob
            medical_profile.sex = sex
            medical_profile.allergies = allergies
            medical_profile.medications = medications
            medical_profile.emergency_contact_name = emergency_contact_name
            medical_profile.emergency_contact_phone = emergency_contact_phone
            medical_profile.blood_type = blood_type
            medical_profile.smoking_status = smoking_status

        db.session.add(medical_profile)
        db.session.commit()

        return redirect(url_for('home'))

    return render_template('home.html', medical_profile=medical_profile)


@app.route('/chat')
@login_required
def chat():
    user = current_user
    # Assuming system messages use sender_id or recipient_id as 0 to represent the server/system
    chat_history = Message.query.filter(
        ((Message.sender_id == user.id) & (Message.recipient_id == 0)) | 
        ((Message.recipient_id == user.id) & (Message.sender_id == 0))
    ).order_by(Message.timestamp.desc()).all()

    for message in chat_history:
        # Replace new lines with HTML line breaks for display in HTML
        message.content = message.content.replace('\\n', '<br>')

    # If no chat history exists, create an initial system message
    if not chat_history:
        system_message = Message(
            content="Hello, I will chat with you about your diagnosis or until you contact a dr. Ask me anything to begin!",
            sender_id=0,  # System as the sender
            sender_type='system',
            recipient_id=user.id,
            recipient_type='user'  # Assuming 'user' type for simplicity
        )
        db.session.add(system_message)
        db.session.commit()
        chat_history = [system_message]

    print(chat_history)
    chat_history = reversed(chat_history)
    return render_template('chat.html', chat_history=chat_history)

@app.route('/submit-message', methods=['POST'])
@login_required
def submit_message():
    message_content = request.form['message']
    user = current_user

    # Assuming '0' for the server/system's ID and that system/server messages are managed as 'system'
    # Creating a message from the user to the system
    user_message = Message(
        content=message_content,
        sender_id=user.id,
        sender_type='user',  # Assuming the current user is a regular user; adjust if it can be a 'doctor'
        recipient_id=0,  # Server ID
        recipient_type='system'
    )
    db.session.add(user_message)
    db.session.commit()

    # Getting a response from the system (simulate conversation)
    response = run_conversation(prompt=message_content)

    # Creating a system response message to the user
    system_response = Message(
        content=response,
        sender_id=0,  # Server as the sender
        sender_type='system',
        recipient_id=user.id,
        recipient_type='user'  # Assuming the recipient is a regular user; adjust if it can be a 'doctor'
    )
    db.session.add(system_response)
    db.session.commit()

    return redirect(url_for('chat'))

@app.route('/chat-message', methods=['POST'])
@login_required
def chat_message():
    message_content = request.form.get('message')
    recipient_username = request.form.get('drname')
    
    print(recipient_username)

    recipient = Doctor.query.filter_by(username=recipient_username).first()

    if not recipient:
        flash('Doctor not found!', 'error')
        return render_template('mcq.html', chat_history=[], drname=recipient_username)

    if message_content != "":
        new_message = Message(
            content=message_content,
            sender_id=current_user.id,
            sender_type='doctor' if isinstance(current_user, Doctor) else 'user',
            recipient_id=recipient.id,
            recipient_type='doctor'
        )
        db.session.add(new_message)
        db.session.commit()

    chat_history = Message.query.filter(
        db.or_(
            db.and_(Message.sender_id == current_user.id, Message.recipient_id == recipient.id),
            db.and_(Message.recipient_id == current_user.id, Message.sender_id == recipient.id)
        )
    ).order_by(Message.timestamp.desc()).all()

    return render_template('mcq.html', chat_history=reversed(chat_history), drname=recipient_username)


@app.route('/mcq', methods=['GET'])
@login_required
def mcq():
    drname = request.args.get('drname', '')  # Get the doctor's name from query parameters
    doctor = Doctor.query.filter_by(username=drname).first()

    if not doctor:
        flash('Doctor not found!', 'error')
        return render_template('mcq.html', chat_history=[], drname=drname)

    # Filter messages by current user and the selected doctor
    chat_history = Message.query.filter(
        db.or_(
            db.and_(Message.sender_id == current_user.id, Message.recipient_id == doctor.id, Message.recipient_type == 'doctor'),
            db.and_(Message.recipient_id == current_user.id, Message.sender_id == doctor.id, Message.sender_type == 'doctor')
        )
    ).order_by(Message.timestamp.desc()).all()

    return render_template('mcq.html', chat_history=reversed(chat_history), drname=drname)


@app.route('/qst', methods=['POST', 'GET'])
@login_required
def qst():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms[]')
        print(symptoms)
        diagnosis = test(symptoms)
        return render_template('qst.html', diagnosis=diagnosis)
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
        type = request.form['type']
        
        if type == 'Patient':
            user = User.query.filter_by(username=username).first()
        elif type == 'Doctor':
            user = Doctor.query.filter_by(username=username).first()
        else:
            flash('Invalid user type specified.', 'danger')
            return render_template('login.html')

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            if type == 'Doctor':
                return redirect(url_for('dr_index'))
            return redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check your username and password.', 'danger')

    return render_template('login.html')

@login_manager.user_loader
def load_user(user_id):
    # Attempt to fetch from both user tables if needed. Adjust according to your app's logic.
    user = User.query.get(int(user_id))
    if user is None:
        user = Doctor.query.get(int(user_id))
    return user

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
        type = request.form['type']

        print(f"I am the server, i got: [{username}, {email}, {password}, {type}]")

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        if type == "Patient":
            existing_user_username = User.query.filter_by(username=username).first()
            existing_user_email = User.query.filter_by(email=email).first()

            print(f"I am the server, im trying to create this user: [existing_user_username = {existing_user_username}, hashed_password = {hashed_password}]")

            if existing_user_username:
                flash('Username is already taken. Please choose a different one.', 'danger')
            elif existing_user_email:
                flash('Email is already taken. Please choose a different one.', 'danger')
            else:
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                user = User(username=username, email=email, password=hashed_password, imgpath="assets/img/avataaars.svg", bio="Enter bio in the Profile Section...")
                db.session.add(user)
                db.session.commit()

                print(f"Username: {username}, HashedPassword: {hashed_password}, Password: {password}, email: {email}")

                flash('Your account has been created! You can now log in.', 'success')
                return redirect(url_for('login'))
            
        elif type == 'Doctor':
            existing_user_username = Doctor.query.filter_by(username=username).first()
            existing_user_email = Doctor.query.filter_by(email=email).first()

            if existing_user_username:
                flash('Username is already taken. Please choose a different one.', 'danger')
            elif existing_user_email:
                flash('Email is already taken. Please choose a different one.', 'danger')
            else:
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                doctor = Doctor(username=username, email=email, password=hashed_password, imgpath="assets/img/avataaars.svg", bio="Enter bio in the Profile Section...")
                db.session.add(doctor)
                db.session.commit()

                print(f"Doctor created: [Username: {doctor.username}, Email: {doctor.email}, Hashed Password: {hashed_password}]")

                flash('Your account has been created! You can now log in.', 'success')
                return redirect(url_for('login'))
            
    return render_template('sign-up.html')



# ======================================================
# DR STUFF


LISENSE_UPLOAD_FOLDER = 'app/static/licenses'
# medical-website/app/static/licenses/
LISENSE_ALLOWED_EXTENSIONS = ['odt', 'doc', 'docx', 'pdf', 'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'svg', 'webp']

def allowed_file_license(filename):
    file_ext = filename.rsplit('.', 1)[1].lower()
    return '.' in filename and file_ext in LISENSE_ALLOWED_EXTENSIONS

def generate_filename_license(username, extension):
    unique_string = f"{username}{time.time()}"
    hashed_string = hashlib.sha256(unique_string.encode()).hexdigest()
    return f"{hashed_string}.{extension}"

@app.route('/dr/')
@login_required
def dr_index():
    username = current_user.username
    imgpath = current_user.imgpath
    bio = current_user.bio
    return render_template('index-drs.html', username=username, imgpath=imgpath, bio=bio)


@app.route('/dr/qst', methods=['POST', 'GET'])
@login_required
def dr_qst():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms[]')
        print(symptoms)
        diagnosis = test(symptoms)
        return render_template('qst-dr.html', diagnosis=diagnosis)
    return render_template('qst-dr.html')


@app.route('/dr/home', methods=['GET'])
@login_required
def dr_home():
    user = current_user
    license = user.lisence # it is a path to a pdf that i will embed in the html
    # path = 'licenses/' + license
    print(license)
    return render_template('home-dr.html', path=license)


@app.route('/dr/profile', methods=['GET', 'POST'])
@login_required
def dr_profile():
    print(current_user)
    imgpath = current_user.imgpath
    if request.method == 'POST':
        username = request.form.get('username').strip()
        bio = request.form.get('bio').strip()
        profession = request.form.get('profession').strip()
        license=''

        # Handle profile picture upload
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and allowed_file(file.filename):
                extension = file.filename.rsplit('.', 1)[1].lower()
                filename = generate_filename(username, extension)
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                os.makedirs(os.path.dirname(upload_path), exist_ok=True)  # Ensure directory exists
                file.save(upload_path)
                imgpath = f'assets/img/profilepics/{filename}'
                if current_user.imgpath and current_user.imgpath != imgpath:
                    old_img_path = os.path.join(UPLOAD_FOLDER, current_user.imgpath.split('/')[-1])
                    if os.path.exists(old_img_path):
                        os.remove(old_img_path)

        # Handle license file upload
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            print(file)
            print(file and allowed_file_license(file.filename))
            if file and allowed_file_license(file.filename):
                extension = file.filename.rsplit('.', 1)[1].lower()
                print(extension)
                filename = generate_filename_license(username, extension)
                license_path = os.path.join(LISENSE_UPLOAD_FOLDER, filename)
                print(license_path)
                os.makedirs(os.path.dirname(license_path), exist_ok=True)  # Ensure directory exists
                file.save(license_path)
                license = f'licenses/{filename}'
                print(license)
                if current_user.lisence and current_user.lisence != license:
                    old_pdf_path = os.path.join(LISENSE_UPLOAD_FOLDER, current_user.lisence.split('/')[-1])
                    if os.path.exists(old_pdf_path):
                        os.remove(old_pdf_path)

        # Update user attributes
        if username != "" and username != current_user.username:
            current_user.username = username
        if bio and bio != current_user.bio:
            current_user.bio = bio
        if profession and profession != current_user.profession:
            current_user.profession = profession
        if imgpath and imgpath != current_user.imgpath:
            current_user.imgpath = imgpath
        if license and license != current_user.lisence:
            current_user.lisence = license

        db.session.commit()
        return redirect(url_for('dr_profile'))

    return render_template('profile-dr.html', imgpath=imgpath)


@app.route('/dr/mcq')
@login_required
def dr_mcq():
    user = current_user
    chat_history = Message.query.filter_by(sender=user).all()
    for message in chat_history:
        if message.message_type == 'user':
            print("User:", message.content)
        else:
            print("Other:", message.content)
        message.content = message.content.replace('\\n', '<br>')
    return render_template('chat-dr.html', chat_history=reversed(chat_history))