from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model (update the path if necessary)
model = joblib.load('ticket_classifier_model1.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        description = request.form.get('description')
        subject = request.form.get('subject')
        # Combine the fields as done during training
        combined_text = description + " " + subject
        pred = model.predict([combined_text])
        prediction = {
            'Category': pred[0][0],
            'Sub-Category': pred[0][1],
            'Group': pred[0][2]
        }
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
