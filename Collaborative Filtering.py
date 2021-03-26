#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.collab import *
from fastai.tabular.all import *
path = untar_data(URLs.ML_100k)


# In[2]:


#Pull movie ratings ratings from MovieLens dataset and put in pandas dataframe
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()


# In[3]:


#Find movie titles
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()


# In[4]:


#Merge the two tables to be able to see the movie titles
ratings = ratings.merge(movies)
ratings.head()


# In[5]:


#Create dataloader
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()


# In[6]:


embs = get_emb_sz(dls)
embs


# In[7]:


class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)


# In[8]:


model = CollabNN(*embs)


# In[9]:


#Running the model using fastai's learner functions, calculating loss as MSE
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)


# In[10]:


#Adding two hidden layers of sizes 100 and 50 nodes respectively
learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)

