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