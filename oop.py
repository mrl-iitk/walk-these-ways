# Tutorial to understand Object Oriented Programming
# Reference - https://youtu.be/JeznW_7DlB0?si=ifbWmxnWoYeZYNXr
import torch
class Cat:
    def __init__(self,name, age): # instantiation method
        self.name = name # creates an attribute
        self.age = age 

    def mult5(self,x): # custom method. Note : self is always
                       # the first param, as an invisible reference
                       # to the created instance
        return x*5
    def speak(self): # by default self is passed when calling a method, 
                    # so you need to explicitly provide the "self" argument
        print("meow")

#INHERITANCE

# Upper level class
class Pet():
    def __init__(self,name, age): 
        self.name = name 
        self.age = age

    def show(self):
        print(f"I am {self.name} and I am {self.age} years old.")

# Lower level class
class Dog(Pet): # inherits from Pet, i.e has same functionalities as in Pet and more
    def speak(self):
        print(f"{self.name} : bark!")

# example of super function
class Fish(Pet):
    def __init__(self, name, age, colour):
        super().__init__(name, age) # calls init of parent class using super
        self.colour = colour # add whatever instantiation you want
    def show(self):
        print(f"I am {self.name} and I am {self.age} years old and of {self.colour} colour.")

p = Pet("Tim",19)
d = Dog("Bill",15)
f = Fish("Jim", 16, "red")
p.show()
d.show()
d.speak()
f.show()

reference_trajectory = torch.load("reference_trajectory.pt")
print(reference_trajectory.keys())