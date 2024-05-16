from __future__ import division, print_function
from flask import Flask, render_template, request,session,logging,flash,url_for,redirect,jsonify,Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_mail import Mail
import os
import secrets
import json
import pickle

from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__,template_folder='template')
app.secret_key = 'super-secret-key'

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = params['gmail_user']
app.config['MAIL_PASSWORD'] = params['gmail_password']
mail = Mail(app)

if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)




class Contact(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    phone_num = db.Column(db.String(12), nullable=False)
    message = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)


class Register(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    rno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(12), nullable=False)
    password2 = db.Column(db.String(120), nullable=False)

@app.route("/")
def home():
    return render_template('index.html',params=params)
  
@app.route("/about")
def about():
    return render_template('about.html',params=params)  


@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    sendmessage=""
    errormessage=""
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('contact')
        message = request.form.get('message')
        try:
            entry = Contact(name=name, phone_num= phone, message = message, email = email,date= datetime.now() )
            db.session.add(entry)
            sendmessage="Thank you for contacting us !.Your message has been sent."
        except Exception as e:
            errormessage="Error : "+ str(e)
        finally:
             db.session.commit()


    return render_template('contact.html',params=params ,sendmessage=sendmessage,errormessage=errormessage)


@app.route("/register", methods=['GET','POST'])
def register():
    if(request.method=='POST'):
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password2')
        error=""
        avilable_email= Register.query.filter_by(email=email).first()
        if avilable_email:
            error="email is already exists"
        else:
            if(password==password2):
                entry = Register(name=name,email=email,password=password, password2=password2)
                db.session.add(entry)
                db.session.commit()
                return redirect(url_for('login'))
            else:
                flash("plz enter right password")
        return render_template('register.html',params=params, error=error)
    return render_template('register.html', params=params)

@app.route("/login", methods=['GET','POST'])  
def login():
    if('email' in session and session['email']):
        return render_template('cyberbullying.html',params=params)

    if (request.method== "POST"):
        email = request.form["email"]
        password = request.form["password"]
        
        login = Register.query.filter_by(email=email, password=password).first()
        if login is not None:
            session['email']=email
            return render_template('cyberbullying.html',params=params)
        else:
            flash("plz enter right password")
    return render_template('login.html',params=params)


@app.route('/', methods=['GET'])
def captiondashboard():
    return render_template('cyberbullying.html', params=params)


# generate a description for an image

@app.route("/logout", methods = ['GET','POST'])
def logout():
    session.pop('email')
    return redirect(url_for('home'))


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


def clean_text(sentence):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    # https://gist.github.com/sebleier/554280
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()
###################################################


# @app.route('/')
# def hello_world():
#     return 'Hello World!'


@app.route('/cyberbullying')
def cyberbullying():
    return flask.render_template('cyberbullying.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    to_predict_list = request.form.to_dict()
    print(to_predict_list)
    review_text = clean_text(to_predict_list['review_text'])
    print(review_text)
    pred = clf.predict(count_vect.transform([review_text]))
    preds=pred[0]
    # print(preds)
    # if pred[0]:
    #     prediction ="Non-Bully"
    # else:
    #     prediction ="Bully"

    return render_template('cyberbullying.html', params=params, prediction=preds)


    # return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(
        debug=False, threaded = False,
        port=5000
    )

