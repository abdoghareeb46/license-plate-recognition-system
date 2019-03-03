import json
import os , sys
from flask import Flask,jsonify,request,render_template
import model as mod

app = Flask(__name__)

@app.route("/Process",methods=["POST"])
def get_res():
    file=request.files['file']
    file.save(os.path.join("/seg/img", file.filename))
    path = os.path.join("/seg/img", file.filename)
    img= mod.detect_plate(path)
    imgList= mod.segment(img)
    mod.classify_noise(imgList)
    #lis=['7', '4', '0', '3', 'R', 'U', 'A'] # Test Function
    #cl=mod.classification(imgList)
    #pn=mod.get_pn(lis)
    check=False
    data={'path':path,"imges":imgList,'res':check}
    return jsonify(data)
#################################### For solving cross ##########################
@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

###################################  Runnting the server #################################################
if __name__ == '__main__':
    app.run(host="127.0.0.1",port=9090)
