from flask import Flask
from markupsafe import escape
from flask import url_for

# 实例化一个对象
app = Flask(__name__)

# 注册一个请求处理函数，通过装饰器app.route()来为这个函数绑定一个url，当用户访问这个url的时候，就会触发这个函数，获取返回值，并将返回值显示到浏览窗口
""" Web程序的理解： 编写不同的函数，处理对应url的请求 """
# 这里的地址是相对地址："/" --> "http://localhost:5000/" ; "/hello" --> "http://localhost:5000/hello"
@app.route('/hello')
@app.route('/')
def hello():
    return '<h1>Welcome to My Watchlist!</h1><img src="http://helloflask.com/totoro.gif">'


@app.route('/user/<name>')
def user_page(name):
    return f"user: {escape(name)}"

