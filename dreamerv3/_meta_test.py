import urllib.request, json
print(urllib.request.urlopen('http://127.0.0.1:8765/meta.json?t=1', timeout=5).read().decode())
