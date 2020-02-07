import gc

class Employee(object):
  Employee_Count = 0
  def __init__(self,name,family,department):
      self.name = name
      self.family = family
      self.department = department
      Employee.increment_employee()
  @property
  def salary(self):
    return self.salary
  @staticmethod    
  def avgSalary():
    total = 0
    employees = 0
    for emp in gc.get_objects():
      if isinstance(emp, Employee):      
        total = emp.salary + total
        employees +=1
    return total / employees
  @staticmethod  
  def get_employee_count():
    return Employee.Employee_Count
  @classmethod  
  def increment_employee(cls):
    cls.Employee_Count =+1

class FullTimeEmployee(Employee):
  def __init__(self, name: str, family: str, department: str, salary: float):
        super().__init__(name, family, department)
        self.salary = salary
  @property
  def salary(self):
        return self._salary
  @salary.setter
  def salary(self, value):
      self._salary = value
  @staticmethod
  def get_average_salary():
      total_salary = 0
      ft_emp_count = 0
      for inst in gc.get_objects():
          if isinstance(inst, FullTimeEmployee):
              total_salary += inst.salary
              ft_emp_count += 1
      return total_salary / ft_emp_count
emp1 = Employee("Landon", "volkmann", "IT")
emp2 = Employee("Trevor", "klinkenberg", "IT")

print(Employee.get_employee_count())

ft_emp_1 = FullTimeEmployee("Dougy", "smith", "synergy", 70000.00)
ft_emp_2 = FullTimeEmployee("Bobby", "burg", "kitchen", 30000.00)

print(Employee.get_employee_count())
print(FullTimeEmployee.get_average_salary())


import requests
from bs4 import BeautifulSoup
import urllib.request
import os

def find_title(url):
  html = requests.get(url)
  bsObj = BeautifulSoup(html.content,"html.parser")
  print(bsObj.title)
  contents = """
  title: {}
  links: {}
  """.format(bsObj.h1, [link.get("href") for link in bsObj.find_all("a")])
  return contents


out = open("output.txt","w")
out.write(str(find_title("https://en.wikipedia.org/wiki/Deep_learning")))
out.close()

import numpy as np

tnA = np.random.randint(1,20,15)
print(tnA)
print("\n")
tnB = tnA.reshape(3,5)
print(tnB)
print("\n")
tnmax =  np.max(tnB,axis=1).reshape(-1, 1)
tnC= np.where(tnB == tnmax,0,tnB)
print(tnC)