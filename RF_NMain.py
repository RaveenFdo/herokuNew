from flask import Flask, request, jsonify
import pickle
#import RF_NEW as rf
from RF_NEW import predict_mpg as rf

app = Flask('Pred')

@app.route('/', methods=['POST'])
def predict():
    vari1 = request.get_json()
    with open("model.pkl", "rb") as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = rf(vari1, model)

    response = {
        'Pred' : int(predictions)
    }
    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)
