from flask import render_template, request, session
from app import app, host, port, user, passwd, db
from app.helpers.database import con_db
import pymysql
from setlist_1 import beatspl2tracks, beats2echonest, EN_id2summary, DiGraph

# To create a database connection, add the following
# within your view functions:
# con = con_db(host, port, user, passwd, db)

myvariables =[]
# ROUTING/VIEW FUNCTIONS
@app.route('/')
@app.route('/index')
def index():
    # Renders index.html.
    session['counter']=-1

    return render_template('index.html')

@app.route('/getplaylistinfo', methods=['POST'])
def getplaylistinfo():
    if session['counter']==-1:
        request.form['query']   
        query = request.form['query'] 
        
        beatstracks = beatspl2tracks(query)
        entracks = beats2echonest(beatstracks)
        summarydf, distance_matrix = EN_id2summary(entracks)
        playlist = []
        playlist = summarydf['song'].tolist()
        UTlist, orderlist = DiGraph(distance_matrix, playlist)


        
        myvariables.append(orderlist)
    session['counter']+=1

    return render_template('getplaylistinfo.html',  query=query, playlist=playlist, myvariables=myvariables)

app.secret_key='junk'

@app.route('/generatedistance')
def generatedistance():
    if session['counter']==-1:
        playlist = myvariables

    return render_template('generatedistance.html', myvariables=myvariables)

    
@app.route('/progressbar')
def progressbar():
    request.form['query']   
    query = request.form['query'] 

    beatstracks = beatspl2tracks(query)
    entracks = beats2echonest(beatstracks)

    return render_template('progress.html', query=query, beatstracks = beatstracks, entracks = entracks)

@app.route('/home')
def home():
    # Renders home.html.
    return render_template('home.html')

@app.route('/slides')
def about():
    # Renders slides.html.
    return render_template('slides.html')

@app.route('/author')
def contact():
    # Renders author.html.
    return render_template('author.html')

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
