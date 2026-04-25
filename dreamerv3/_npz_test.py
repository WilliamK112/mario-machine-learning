import glob, os, numpy as np, hashlib
log='c:/Users/31660/mario-rl/dreamerv3/logs/mario_run3_rescue'
files=glob.glob(log+'/eval_eps/*.npz')+glob.glob(log+'/train_eps/*.npz')
f=max(files,key=os.path.getmtime)
d=np.load(f)
frames=d['image']
print('file',os.path.basename(f),'len',len(frames))
for i in [0,1,2,10,50,min(100,len(frames)-1)]:
    m=hashlib.md5(frames[i].tobytes()).hexdigest()
    print(i,m)
