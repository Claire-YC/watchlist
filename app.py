from flask import Flask, render_template
from markupsafe import escape
from flask import url_for

# 页面填充数据
name = "Yang Cao"
movies = [
    {"title": "My Neighbor Totoro", "year": "1998"},
    {"title": "Dead Poets Society", "year": "1989"},
    {"title": "Call Me by Your name", "year": "2017"},
    {"title": "Oppenheimer", "year": "2023"},
    {"title": "Barbie", "year": "2023"}
]


# 实例化一个对象
app = Flask(__name__)

# 注册一个请求处理函数，通过装饰器app.route()来为这个函数绑定一个url，当用户访问这个url的时候，就会触发这个函数，获取返回值，并将返回值显示到浏览窗口
""" Web程序的理解： 编写不同的函数，处理对应url的请求 """
# 这里的地址是相对地址："/" --> "http://localhost:5000/" ; "/hello" --> "http://localhost:5000/hello"
@app.route('/hello')
def hello():
    return '<h1>Welcome to My Watchlist!</h1><img src="http://helloflask.com/totoro.gif">'

@app.route("/")
def index():
    return render_template('index.html', name=name, movies=movies)


@app.route('/user/<name>')
def user_page(name):
    return f"user: {escape(name)}"

