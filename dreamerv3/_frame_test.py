import urllib.request, hashlib
u='http://127.0.0.1:8765/frame.jpg'
b1=urllib.request.urlopen(u+'?i=0&t=1', timeout=5).read()
b2=urllib.request.urlopen(u+'?i=10&t=2', timeout=5).read()
print('len1',len(b1),'len2',len(b2))
print('same', b1==b2)
print('md5-1',hashlib.md5(b1).hexdigest())
print('md5-2',hashlib.md5(b2).hexdigest())
