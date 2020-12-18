import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Input tweet':'kamu jahat kaya china komunis'})

print(r.json())