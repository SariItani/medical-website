# app/__init__.py
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin

app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = '9d125d217fca3a2938b669e0553189a4'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)


login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    imgpath = db.Column(db.String(255))
    bio = db.Column(db.Text)

    def __init__(self, username, email, password, imgpath=None, bio=None):
        self.username = username
        self.email = email
        self.password = password
        self.imgpath = imgpath
        self.bio = bio
        

class MedicalProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    dob = db.Column(db.Date)
    sex = db.Column(db.Text)

    # Medical History
    allergies = db.Column(db.Text)
    medications = db.Column(db.Text)  # Consider using a separate table for medications with details like dosage and frequency
    conditions = db.Column(db.Text)  # Consider using a separate table for conditions with details like diagnosis date and severity
    immunizations = db.Column(db.Text)  # Consider a separate table for detailed immunization records
    surgical_history = db.Column(db.Text)  # Consider a separate table for details on past surgeries

    # Administrative Information
    emergency_contact_name = db.Column(db.Text)
    emergency_contact_phone = db.Column(db.Text)
    insurance_provider = db.Column(db.Text)
    insurance_id = db.Column(db.Text)
    primary_care_physician_id = db.Column(db.Integer, db.ForeignKey('doctor.id'))
    primary_care_physician = db.relationship('Doctor', backref='patients')


    # Other relevant fields
    blood_type = db.Column(db.Text)
    smoking_status = db.Column(db.Boolean)
    family_history = db.Column(db.Text)

    user = db.relationship('User', backref=db.backref('medical_profile', uselist=False))
        

class Doctor(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    imgpath = db.Column(db.String(255))
    bio = db.Column(db.Text)
    lisence = db.Column(db.String(255))
    profession = db.Column(db.String(50))
    
    def __init__(self, username, email, password, imgpath=None, bio=None, lisence=None, profession=None):
        self.username = username
        self.email = email
        self.password = password
        self.imgpath = imgpath
        self.bio = bio
        self.lisence = lisence
        self.profession = profession


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(500), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender = db.relationship('User', backref=db.backref('messages', lazy=True))
    message_type = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, content, sender, message_type):
        self.content = content
        self.sender = sender
        self.message_type = message_type


from app import routes
