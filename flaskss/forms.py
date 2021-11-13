from flask_wtf import FlaskForm
from wtforms import PasswordField,StringField,SubmitField,BooleanField#for phone number ,IntegerField
from wtforms.validators import  Email,DataRequired,Length,EqualTo

class RegistrationForm(FlaskForm):
    username=StringField("Username",validators=[DataRequired(),Length(min=2,max=12)])
    email = StringField("Email",validators= [DataRequired(),Email()])
    password=PasswordField("Password",validators=[DataRequired()])
    confirm_password=PasswordField("Confirm password",validators=[DataRequired(),EqualTo("Password")])
    Submit=SubmitField("Sign up")

class LoginForm(FlaskForm):
    email = StringField("Email",validators= [DataRequired(),Email()])
    password=PasswordField("Password",validators=[DataRequired()])
    remember=BooleanField("Remember me")
    submit=SubmitField("Login")