from flask import Flask, render_template, request, session, url_for
from app import app, host, port, user, passwd, db
from app.helpers.database import con_db
import pymysql
from NEVERRESETHEAD import beatspl2tracks, beats2echonest, EN_id2summary, DiGraph
import csv

# To create a database connection, add the following
# within your view functions:
# con = con_db(host, port, user, passwd, db)

# ROUTING/VIEW FUNCTIONS
@app.route('/')
@app.route('/index')
def index():
    # Renders index.html.

    return render_template('index.html')

app.secret_key = 'junk'

@app.route('/getplaylistinfo', methods=['POST'])
def getplaylistinfo():
    

    request.form['query']   
    query = request.form['query'] 
    
    beatstracks = beatspl2tracks(query)
    entracks = beats2echonest(beatstracks)
    filename = 'request_' + query
    songdatalist, dist_matrix, playlist = EN_id2summary(filename, entracks)
    bestlist, minval, shuffle, avg_shuffle, improvement, orig_tups= DiGraph(songdatalist, dist_matrix, playlist)
    


    return render_template('getplaylistinfo.html',  query=query, playlist=playlist)

app.secret_key='junk'

@app.route('/generatedistance', methods=['POST'])
def generatedistance():

    print request.form['query']   
    query = request.form['query'] 
    
    beatstracks = beatspl2tracks(query)
    entracks = beats2echonest(beatstracks)
    filename = 'request_' + query + '.png'
    songdatalist, dist_matrix, playlist = EN_id2summary(filename, entracks)
    bestpath, minval, shuffle, avg_shuffle, improvement, orig_tups = DiGraph(songdatalist, dist_matrix, playlist)
    filename = url_for('static', filename=filename)
    


        
    '''requested_loan_amount = loan_info[0]['loan_amount']
    requested_repayment_term = loan_info[0]['terms_repayment_term']
    binreqLoanAmount=int(round(loan_info[0]['loan_amount'],-2))/100

    if requested_repayment_term > 10:
        month_pred = [i for i in range(requested_repayment_term-10,requested_repayment_term+10)]
    else:
        month_pred = [(i+1) for i in range(20)]

    roundedReqAmt = int(math.floor(requested_loan_amount/100))

    if binreqLoanAmount >= 10:
        gridAmtSet = binreqLoanAmount
    else:
        gridAmtSet = 5

    amount_pred = [100*i for i in range(gridAmtSet-5,gridAmtSet+5)]
    
    #create list of tuples for (amount, month, funding prob )
    predMatrix=[]
    for amount in amount_pred:
        for month in month_pred:
            xin = np.append(continent_vec,sector_vec)
            xin = np.append(xin,np.array( [borrowers_gender,loan_info[0]   ['description_num_languages'],amount,posted_date_months,month ] ))


            predprob = round(clf.predict_proba(xin)[0][0],2)
            predMatrix.append((amount, month, predprob) )'''
    
    # write to tsv file
    ofile  = open('app/static/data.tsv', "wb")
    writer = csv.writer(ofile,delimiter='\t')

    writer.writerow(['pre','post', 'weight'])
    for song in orig_tups:
                writer.writerow(song)
    ofile.close()

    '''amount_pred_str = str(amount_pred)
    month_pred_str = str(month_pred)'''
    return render_template('generatedistance.html', query=query, url = filename, bestpath=bestpath, minval=minval, shuffle=shuffle, avg_shuffle=avg_shuffle, improvement=improvement, tups = orig_tups)
    
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
