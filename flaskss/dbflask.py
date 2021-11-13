from main_flask import db
from main_flask import User,Post
#db.create_all()
#Add data to site.db
user_2=User(username="ameya",email="ameyagidh2@gmail.com",password="asd")
db.session.add(user_2)
db.session.commit()
User.query.all()
User.query.first()
user=User.query.filter_by(username="ameya").all()
user.id
u=User.query.get(1)
user.posts
user_id=User.id(1)
post_1=Post(title="blog 1",content="hello my first post",user_id=user_id)
post_1.author#foreign backref
db.drop_all()