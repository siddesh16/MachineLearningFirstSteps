import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff


# Importing dataset and examining it
dataset = pd.read_csv("Titanic.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')

# Converting Categorical features into Numerical features
dataset['Sex'] = dataset['Sex'].map({'male':1, 'female':0})

# Defining Child & Adult
def converter(column):
    if column <= 13:
        return 1 # Child
    else:
        return 0 # Adult

dataset['Age'] = dataset['Age'].apply(converter)
print(dataset.head())
print(dataset.info())

# Dividing dataset into label and feature sets
X = dataset.drop(['Survived','Embarked', 'Fare'], axis = 1) # Features
Y = dataset['Survived'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Implementing PCA to visualize dataset
pca = PCA(n_components = 2)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))

survival= list(dataset['Survived'])
age = list(dataset['Age'])
sex = list(dataset['Sex'])
pclass = list(dataset['Pclass'])
data = [go.Scatter(x=x_pca[:,0], y=x_pca[:,1], mode='markers',
                    marker = dict(color=Y, colorscale='Rainbow', opacity=0.5),
                                text=[f'Survived: {a}; Age: {b}; Sex:{c}, Pclass:{d}' for a,b,c,d in list(zip(survival,age,sex,pclass))],
                                hoverinfo='text')]

layout = go.Layout(title = 'PCA Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Principal Component'),
                    yaxis = dict(title='Second Principal Component'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='pca.html')

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity = 5,n_iter=2000)
x_tsne = tsne.fit_transform(X_scaled)

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=Y, colorscale='Rainbow', opacity=0.5),
                                text=[f'Survived: {a}; Age: {b}; Sex:{c}, Pclass:{d}' for a,b,c,d in list(zip(survival,age,sex,pclass))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE.html')
