from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the file to a temporary location
            temp_file_path = os.path.join('uploads', file.filename)
            file.save(temp_file_path)
            return render_template('index.html', message='File uploaded successfully!', filename=file.filename)
    return render_template('index.html')

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)