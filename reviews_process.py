import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import regex
import emoji
from pycountry import languages
import fasttext

def split_count(info):

    emoji_list = []
    data = regex.findall(r'\X', info)
    for word in data:
        if any(char in emoji.EMOJI_DATA for char in word):
            emoji_list.append(word)

    return len(emoji_list)


reviews_raw = pd.read_csv("reviews.csv")
PRETRAINED_MODEL_PATH = 'lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

reviews_raw['lang'] = range(0, len(reviews_raw))

for index, row in reviews_raw.iterrows():
    if type(row['comments']) == type('s'):
      predictions = model.predict(row['comments'])
      l = predictions[0][0].split('_label_')[1]
      if l != 'ceb' and l != 'nds' and l != 'war' and l != 'wuu':
          reviews_raw['lang'][index] = l

reviews_raw = reviews_raw[reviews_raw['comments'].notna()]

reviews_raw['emoji_count'] = reviews_raw.comments.apply(split_count)

reviews_raw['lenstr'] = reviews_raw['comments'].str.len()

reviews_raw['test_emoji'] = (((reviews_raw['emoji_count'] == reviews_raw['lenstr']) |
                          (2*reviews_raw['emoji_count'] == reviews_raw['lenstr'])) & (reviews_raw['emoji_count']!=0))

reviews_raw = reviews_raw[reviews_raw.test_emoji == False]

reviews_raw = reviews_raw.drop(columns='test_emoji')
reviews_raw = reviews_raw.drop(columns='lenstr')
reviews_raw = reviews_raw.drop(columns='emoji_count')

reviews_raw.comments = reviews_raw.comments.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
reviews_raw['emoji_count'] = reviews_raw.comments.apply(split_count)

reviews_raw = reviews_raw[(reviews_raw.comments.str.len() > 20)]
reviews_raw['comments'] = reviews_raw['comments'].str.replace('<br/>', '')

reviews_raw = reviews_raw[reviews_raw.lang == '_en']

reviews_raw.to_csv("reviews_processed.csv")


reviews = pd.read_csv("reviews_processed.csv", lineterminator='\n')

print("Mean: ", reviews['comments'].str.split().str.len().mean())
print("Median: ", reviews['comments'].str.split().str.len().median())
i = sns.kdeplot(reviews['comments'].str.split().str.len(),color="b", label='Words in each Comment')
xi = i.lines[0].get_xdata()
yi = i.lines[0].get_ydata()
meani = reviews['comments'].str.split().str.len().mean()
mediani = reviews['comments'].str.split().str.len().median()
heighti = np.interp(meani, xi, yi)
heighti2 = np.interp(mediani, xi, yi)
i.vlines(meani, 0, heighti, color='r', ls=':', label='Mean')
i.vlines(mediani, 0, heighti2, color='g', ls=':', label='Median')
plt.legend()
plt.show()

stop = stopwords.words('english')
reviews['comments_stp_rem'] = reviews['comments'].apply(lambda x: ' '.join([word.lower() for word in x.split() 
                                                                 if word not in (stop)]))

tr_idf_model  = TfidfVectorizer(analyzer = 'word', max_features=33)
tf_idf_vector = tr_idf_model.fit_transform(reviews.comments_stp_rem)

tf_idf_array = tf_idf_vector.toarray()
words_set = tr_idf_model.get_feature_names_out()
df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
df_tf_idf['listing_id'] = reviews.listing_id

test = df_tf_idf.groupby('listing_id').size()
review_1 = pd.DataFrame({'listing_id':test.index, 'num_cmt':test.values})

test2 = pd.DataFrame([])
for i in words_set:
        test2[f'term_{i}'] = df_tf_idf.groupby(['listing_id'])[f'{i}'].mean()

review_final = review_1.merge(test2, on='listing_id')

review_final.to_csv("reviews_final.csv")