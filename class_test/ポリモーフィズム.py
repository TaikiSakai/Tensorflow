from abc import ABC, abstractclassmethod

#抽象メソッドはメソッドの定義を強制できる
class Greet(ABC):

    @abstractclassmethod
    def greeting(self):
         pass
    
class English(Greet):
    
    def greeting(self):
        print("Hello")

class French(Greet):
    
    def greeting(self):
        print("Bonjour")

class Spanish(Greet):
    
    def greeting(self):
        print("Hola")


fr = French()
sp = Spanish()

fr.greeting()
sp.greeting()



print("-------------------------------")

class Worker(ABC):

    def __init__(self, name):
        self.name = name

    def hello(self):
        print("My name is {}".format(self.name))

    @abstractclassmethod
    def do_work(self):
        pass

class Teacher(Worker):

    def do_work(self):
        print(self.name + " is a teacher")

class Programmer(Worker):

    def do_work(self):
        print(self.name + " is a coder")
    
def instract(worker):
    worker.do_work()

teacher = Teacher("same")
coder = Programmer("sakai")

teacher.do_work()
coder.do_work()
print("------------------------------------")
instract(teacher)
instract(coder)