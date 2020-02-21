import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
train = pd.read_csv('train.csv')
##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
garage = train.get('GarageArea')
z = np.abs(stats.zscore(garage))
print(z)
outliers=[]
drop =  []
for i, score in enumerate(z):
  if score > 3:
    outliers.append((i,score))
    drop.append(i)
print(outliers)
train.drop(train.index[drop],inplace=True)

##visualize

plt.scatter(train.get('GarageArea'), train.get('SalePrice'), alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Relation')
plt.show()
