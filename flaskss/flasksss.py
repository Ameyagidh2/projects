
from flask import Flask,render_template
app=Flask(__name__)
@app.route("/")
def hello():

    return render_template("index.html")


@app.route("/about")
def ameya():
    name_py = "ameya"
    return render_template("about.html",name_temp=name_py)

@app.route("/bootstrap")
def bootstrap():
    name_py = "ameya"
    return render_template("bootstrap.html")

app.run(debug=True)