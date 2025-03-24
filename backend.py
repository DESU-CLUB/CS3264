import flask
from flask import request, jsonify
import pandas as pd #Use this if you have no GPU

app = flask.Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        # Read the CSV using cudf
        try:
            df = pd.read_csv(file)
            # Return basic info about the dataframe
            return jsonify({
                'success': True,
                'rows': len(df),
                'columns': df.columns.tolist()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File must be a CSV'}), 400

if __name__ == '__main__':
    app.run(debug=True)
