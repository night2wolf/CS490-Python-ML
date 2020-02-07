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