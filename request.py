import requests

url = 'http://localhost:5000/predict'
r = requests.post(url,json={'Journey_day':10, 'Journey_month':5, 'Duration':170 })
print(r.json())