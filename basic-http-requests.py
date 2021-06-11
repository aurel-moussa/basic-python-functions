#Basic http requests in Python, to be used for web scraping purposes in later projects

#import libraries
import requests
import os 
from PIL import Image
from IPython.display import IFrame

#assinging URL and request contents
url='https://plato.stanford.edu/'
r=requests.get(url)

#check if response was good
r.status_code
print(r.request.headers)
print("request body:", r.request.body)
header=r.headers
print(r.headers)
header['date']
header['Content-Type']
r.text[0:500]

######getting individual images instead#######
#assinging URL and request contents
url='https://gitlab.com/ibm/skills-network/courses/placeholder101/-/raw/master/labs/module%201/images/IDSNlogo.png'
r=requests.get(url)


#check if response was good
r.status_code
print(r.headers)
r.headers['Content-Type']

#specifiying where to save the image i.e. in current working directory, and as image.png
path=os.path.join(os.getcwd(),'image.png')
path

#saving it, using with open, to auto-close the file afterwards, set to (w)rite in (b)inary mode for images
with open(path,'wb') as f:
    f.write(r.content)

#using the PIL Image library to check whether everything looks fine
Image.open(path)   

#######sending API requests via HTML and JSON payloads (GET) #######

#specifing HTML location and request type
url_get='http://httpbin.org/get'

#specifing payload
payload={"name":"Joseph","ID":"123"}

#requesting
r=requests.get(url_get,params=payload)

#checking if everything worked out ok
print("request body:", r.request.body)
print(r.status_code)
print(r.text)
r.headers['Content-Type']

#converting the json into Python dictionary
r.json()
r.json()['args']

#######sending API requests via HTML and JSON payloads (POST)#######
url_post='http://httpbin.org/post'
payload={"name":"Klaus-Dieter","ID":"123123123"}

#requesting post
r_post=requests.post(url_post,data=payload)
