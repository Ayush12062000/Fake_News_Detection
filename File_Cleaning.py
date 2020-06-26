#%%
import pandas as pd
import warnings
DeprecationWarning("ignore")
warnings.filterwarnings("ignore")

#%%
os.chdir('C:/Users/Ayush/Desktop/Fake_news')
os.listdir()

#%%
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# %%
true["fake_news"] = 0

#%%
fake["fake_news"] = 1

# %%
fake["subject"].unique()

# %%
true["subject"].unique()

# %%
just_text = true["text"]
just_text.head()

# %%
just_text = just_text.str.extractall(r"^.* - (?P<text>.*)")

# %%
just_text = just_text.droplevel(1)

# %%
true = true.assign(text = just_text["text"])

# %%
df = pd.concat([fake , true], axis= 0)

# %%
df  = df.drop(["subject", "title", "date"],axis = 1)

# %%
df.info()

# %%
df.isnull().sum()   #tells total no. of null values

# %%
df = df.dropna(axis = 0)

# %%
clean_text = df.to_csv("cleaned_news.csv",index = False)


# %%
