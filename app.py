from flask import Flask,request, url_for, redirect, render_template,Response
import numpy as np
import joblib
import cv2
from playsound import playsound
vv=0
app = Flask(__name__)
model1=joblib.load('forestfiremodel.pkl')
camera=cv2.VideoCapture(0)
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')
m2=joblib.load('m2.pkl')







@app.route('/')
def hello_world():
    vv=1
    return render_template("index.html")
@app.route('/output')
def output():
    vv=1
    return render_template("index.html")
@app.route('/co')
def co():
    return render_template("co.html")













    
def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        elif(vv==1):
            camera.release()
            cv2.destroyAllWindows()
        else:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
            for (x,y,w,h) in fire:
                cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                playsound('audio.mp3')
               

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()



            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        


@app.route('/wild1')
def wild1():
    vv=1
    return render_template('wild1.html')
model=joblib.load('m1.pkl')

@app.route('/wild',methods=['POST','GET'])
def wild():
    vv=1
    fin=[float(x) for x in request.form.values()]
    fin=[np.array(fin)]
    p=model.predict(fin)
    i=p[0]
    output='{}'.format(i)
    print(output)
    return render_template('output.html',pred="As per the given coditions of the forest , '{}' are in danger  ".format(output))


@app.route('/predict')
def predict1():
    vv=1
    return render_template('fireform.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    vv=1
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model1.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('output.html',pred='Probability of fire occuring is {} \n Your Forest is in Danger'.format(output))
    else:
        return render_template('output.html',pred='Probability of fire occuring is {} \n Your Forest is safe.'.format(output))



@app.route('/detect')
def index():

    return render_template('detect.html')

@app.route('/video')

def video():

    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/burn')
def burn():
    vv=1
    return render_template('burn.html')


@app.route('/burn',methods=['POST','GET'])
def burn1():
    vv=1
    fin=[float(i) for i in request.form.values()]
    fin=[np.array(fin)]
    p=m2.predict(fin)
    i=p[0]
    output='{:.2f}'.format(i)
    print(output)
    return render_template('output.html',pred="As per the given coditions The Predict burn area is '{}'  (ha) hectares  ".format(output))


if __name__ == '__main__':
    app.run(debug=True)
