from flask import  Flask,request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from CNNClassifierProject.utils.common import decode_base64
from CNNClassifierProject.components.predict import Prediction
from CNNClassifierProject.config.configuration import PredictConfig
from CNNClassifierProject.config.configuration import ConfigurationManager
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        configMngr = ConfigurationManager()
        config=configMngr.get_predict_config()
        self.classes={0: 'Coccidiosis', 1: 'Healthy', 2: 'Salmonella', 3: 'New Castle Disease'}
        self.prediction=Prediction(config=config,classes=self.classes)

    
@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train",methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system('python main.py')
    return "Training done successfully"

@app.route("/predict",methods=['POST'])
@cross_origin()
def predictRoute():
    image=request.json['image']
    decode_base64(image,clApp.filename)
    result= clApp.prediction.predict(clApp.filename)
    return jsonify(result)

if __name__ == '__main__':
    clApp=ClientApp()
    app.run(host='0.0.0.0', port=8080)
