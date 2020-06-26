#%%
import pandas as pd
import warnings
DeprecationWarning("ignore")
warnings.filterwarnings("ignore")

#%%
os.chdir('C:/Users/Ayush/Desktop/Fake_news')
os.listdir()

# %%
df = pd.read_csv('cleaned_news.csv')

# %%
DV = "fake_news"
X = df.drop([DV], axis = 1)
y = df[DV]

# %%
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state = 0)

# %%
from sklearn.feature_extraction.text import CountVectorizer

Cv = CountVectorizer(max_features=5000)
X_train_counts = Cv.fit_transform(X_train["text"])
print(Cv.vocabulary)
X_test = Cv.transform(X_test["text"])

# %%
from sklearn.naive_bayes import MultinomialNB

Naive = MultinomialNB()
Naive.fit(X_train_counts , y_train)

# %%
from sklearn.metrics import classification_report , accuracy_score

prediction = Naive.predict(X_test)

print("Score:",accuracy_score(prediction , y_test)*100)

# %%
print("Report:",classification_report(y_test , prediction ))

#%%
def classifier(text):
    Naive = MultinomialNB()
    Naive.fit(X_train_counts, y_train)
    
    word_vec = Cv.transform(text) 
    
    predict = Naive.predict(word_vec)
    return "Fake News Story" if predict[0] else "Real News Story"

# %%
#some real life news
onion = ["""Calling the country singer’s place at the top of Top 200 completely illegitimate, fans of the 
            rapper–singer Drake took to social media Friday to accuse Kenny Chesney of manipulating Billboard’s 
            algorithm by putting effort into his album. “It’s just unfair that this guy could keep Drake from his
            rightful place on the charts by putting out quality music that he actually cares about,” said Aiden 
            Howard, 14, who echoed the sentiments of Drake fans worldwide in his assertion that the artist’s 
            mediocre B-sides deserved more acclaim and recognition. “He clearly gamed the streaming numbers when 
            he decided to put time and energy into his craft. It’s such horseshit that Billboard rewards that 
            behavior and punishes Drizzy for making a half-assed mixtape full of songs he’d already dropped on 
            SoundCloud. How the hell is ‘Toosie Slide’ going to compare to a song that the artist thought about 
            for more than 15 minutes?” At press time, Drake released a statement asking fans to ignore Kenny 
            Chesney and focus on the horseshit that he just released."""]

#THis IS FAKE NEWS LET'S SEE WHAT OUR MODEL PREDICTS

classifier(onion)
# %%
nyt = ["""Two top congressional Democrats opened an investigation on Saturday into President Trump’s removal of 
          Steve A. Linick, who led the office of the inspector general at the State Department, citing a pattern 
          of “politically-motivated firing of inspectors general.” Mr. Trump told Speaker Nancy Pelosi late 
          Friday night that he was ousting Mr. Linick, who was named by President Barack Obama to the State 
          Department post, and replacing him with an ambassador with close ties to Vice President Mike Pence in 
          the latest purge of inspectors general whom Mr. Trump has deemed insufficiently loyal to his 
          administration. In letters to the White House, State Department, and Mr. Linick, Representative Eliot 
          L. Engel of New York, the chairman of the House Foreign Affairs Committee, and Senator Bob Menendez of 
          New Jersey, the top Democrat on the Senate Foreign Relations Committee, requested that the administration
          turn over records and information related to the firing of Mr. Linick as well as “records of all I.G. 
          investigations involving the Office of the Secretary that were open, pending, or incomplete at the 
          time of Mr. Linick’s firing.” Mr. Engel and Mr. Menendez said in their letters that they believe 
          Secretary of State Mike Pompeo recommended Mr. Linick’s ouster because he had opened an investigation 
          into Mr. Pompeo’s conduct. The lawmakers did not provide any more details, but a Democratic aide said 
          that Mr. Linick had been looking into whether Mr. Pompeo had misused a political appointee at the State 
          Department to perform personal tasks for himself and his wife. “Such an action, transparently designed to
          protect Secretary Pompeo from personal accountability, would undermine the foundation of our democratic 
          institutions and may be an illegal act of retaliation,” the lawmakers wrote. Under law, the administration
          must notify Congress 30 days before formally terminating an inspector general. Mr. Linick is expected to 
          leave his post then. Mr. Trump’s decision to remove Mr. Linick is the latest in a series of ousters aimed
          at inspectors general who the president and his allies believe are opposed to his agenda. In May, Mr. 
          Trump moved to oust Christi A. Grimm, the principal deputy inspector general for the Department of Health
          and Human Services, whose office had issued a report revealing the dire state of the nation’s response to
          the pathogen. He has also taken steps to remove two other inspectors general, for the intelligence
          community and for the Defense Department. Mr. Linick was spotlighted during the impeachment inquiry when 
          he requested an urgent meeting with congressional staff members to give them copies of documents related 
          to the State Department and Ukraine, signaling they could be relevant to the House investigation into 
          whether President Trump pressured Ukraine to investigate former Vice President Joseph R. Biden Jr. and 
          his son Hunter Biden. The documents — a record of contacts between Rudolph W. Giuliani, the president’s 
          personal lawyer, and Ukrainian prosecutors, as well as accounts of Ukrainian law enforcement proceedings 
          — turned out to be largely inconsequential."""]

#THIS IS A REAL NEWS LET'S SEE WHAT MODEL PREDICTS

classifier(nyt)


# %%
