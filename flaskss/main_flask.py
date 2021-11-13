from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask,render_template,url_for,flash,redirect
from forms import RegistrationForm,LoginForm

app=Flask(__name__)
'''
posts=[{'name':'Ameya','school':"Podar international","Marks":'9.2 in spce , 81.69 % in 12th and 90.20 % in 10th'},
      {'name':'Leena','school':"St. Jons High School","Marks":'60 in dhanukar , 60 % in 12th and 73 % in 10th'},
     {'name':'Santosh','school':"Fatima devi high school","Marks":'7.5 in spce , 80 % in 12th and 78 % in 10th'}]
import secrets
print(secrets.token_hex(16))
'''
app.config["SQLALCHEMY_DATABASE_URI"]="SQLITE:///SITE.DB"
db=SQLAlchemy(app)

class User(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    username= db.Column(db.String(20),unique=True,nullable=False)
    email = db.Column(db.String(120),unique=True,nullable=False)
    password = db.Column(db.String(30),nullable=False)
    image_file=db.Column(db.String(20),nullable=False,default="default.jpg")
    posts=db.relationship("Post",backref="author",lazy=True)
    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"

class Post(db.Model):
    id= db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    date_posted=db.Column(db.DateTime, nullable=False,default=datetime.utcnow())
    title=db.Column(db.String(40),nullable=False)
    user_id=db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
    def __repr__(self):
        return f"Post('{self.title}','{self.date_posted}')"

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]

#we can use variables in flask using jinga template

app.config["SECRET_KEY"]="7e0a7bdf50dd095369a6a1cce9ef1db2"#fOR SECRECY OF FORMS
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html",posts=posts)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register",methods=["GET","POST"])#Well, according to the standards for HTTP methods, one uses. POST to insert a new object, GET to select one
def register():
    form=RegistrationForm()
    if form.validate_on_submit():
        flash(f"Account created for {form.username.data}!","success")#send this category and data to html
        return redirect(url_for("home"))#function name
    return render_template("register.html",title="Registration form",form=form)

@app.route("/login",methods=["GET","POST"])#Well, according to the standards for HTTP methods, one uses. POST to insert a new object, GET to select one
def login():
    forms=LoginForm()
    if(forms.validate_on_submit()):
      if (forms.email.data=="admin@gmail.com" and forms.password.data=="asd"):
         flash("login successful","success")
         return redirect(url_for('home'))
      else:
          flash("login failed","danger")
    return render_template("login.html",title='Login',forms=forms)

if __name__ == '__main__':
    app.run(debug=True)