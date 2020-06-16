from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

if __name__ == "__main__":
    app.run('127.0.0.1')