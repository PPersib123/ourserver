import matplotlib.pyplot as plt
from Predictor import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# pip install tensorflow
# pip install keras
# pip install sklearn

# Loading model and weight
# load json and create model
json_file = open('./ModelandWeight/Fault_Detection_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./ModelandWeight/Fault_Detection_Weight.h5")
seismic = np.load('seismic2d.npy')
seismic = myGaussSmooth(seismic)
v = np.percentile(seismic, 99)
seismic = clip(seismic, -v, v)
pred1_parahakan = Predictor3(np.transpose(seismic), loaded_model, 64)

q = pred1_parahakan[1]
q = q[:,:,1]
y,x = q.shape
th = 0.5
for i in range(y):
  for j in range(x):
    if q[i,j] > th:
      q[i,j] = 1
    else:
      q[i,j] = 0
q = q.T

y, x = q.shape

xx, yy, zz = [], [], []
for i in range(y):
  for j in range(x):
    if q[i, j] == 1:
      xx.append(j)
      yy.append(i)
      zz.append([j, i])
xx = np.array(xx)
yy = np.array(yy)
zz = np.array(zz)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(zz)
# cluster the data into five clusters
dbscan = DBSCAN(eps=0.2 , min_samples = 1, algorithm='brute')
clusters = dbscan.fit_predict(X_scaled)


dele = []
for i in range(len(clusters)):
  if list(clusters).count(clusters[i]) < 300:
    dele.append(i)
xx = np.delete(xx, dele)
yy = np.delete(yy, dele)
clusters = np.delete(clusters, dele)


seismic = np.load('seismic2d.npy')
clusters = np.full_like(seismic,0)
label = np.full_like(seismic, 0)

for i in range(len(xx)):
  label[yy[i], xx[i]] = 1

fig = plt.figure(constrained_layout=False, figsize=(15,15))
v = np.percentile(seismic, 99)
plt.imshow(seismic,cmap="Greys", vmin = -v, vmax = v)
plt.imshow(label, cmap = 'Reds', alpha=0.41)
plt.show()