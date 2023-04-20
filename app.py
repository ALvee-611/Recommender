import streamlit as st
import sqlite3
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

st.set_page_config(
    page_title='Item Recommender',
    layout="wide"
)

@st.cache_resource
def load_model():
    model_nlp = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model_nlp

# Main NLP model
nlp_model = load_model()


@st.cache_resource
def similar_items(full_df, article_id, num_items = 6):
    """
    takes the dataframe full_df containing the articl_id and all the features and the articl_id of the
    items for which we need similar items, and then return list indices of the num_items(default 5) most similar items
    similar_items: DataFrame Int Int -> Listof(Int)
    """
    # get item index
    item_index = full_df[full_df[246] == article_id].index[0]
    # saving the ids
    id_df = full_df[246]
    # dropping the article_id feature
    main_df = full_df.drop(columns=[246], axis=1)

    item_A = np.array(main_df.iloc[item_index]).reshape(1, -1)

    # NearestNeighbors model(taking 6 since first item will be the item itself)
    nbrs = NearestNeighbors(n_neighbors=num_items, algorithm='auto').fit(main_df)

    # get distances and indices of nearest neighbors
    _, indices = nbrs.kneighbors(item_A)

    # the article_id of 5 most similar items
    id = list(id_df[indices[0]])

    return id[1:]


st.title('Product Store') 
st.text('Welcome to the store!')

## Define all session state variables:
if 'num_items' not in st.session_state:
    st.session_state['num_items'] = 0
if 'article_id' not in st.session_state:
    st.session_state['article_id'] = ""
if 'show_rec' not in st.session_state:
    st.session_state['show_rec'] = False
if 'show_search_items' not in st.session_state:
    st.session_state['show_search_items'] = False
if 'article_liked' not in st.session_state:
    st.session_state['article_liked'] = ""
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'show_popularity' not in st.session_state:
    st.session_state['show_popularity'] = True
if 'show_popular_items' not in st.session_state:
    st.session_state['show_popular_items'] = True
if 'popular_article_id' not in st.session_state:
    st.session_state['popular_article_id'] = ""
if 'similar_descriptions' not in st.session_state:
    st.session_state['similar_descriptions'] = ""
if 'search_article_id' not in st.session_state:
    st.session_state['search_article_id'] = ''
if 'rec_article_id' not in st.session_state:
    st.session_state['rec_article_id'] = ""

@st.cache_resource
def get_products(offset=0):
    conn = sqlite3.connect('data//popularity_V2.db')
    c = conn.cursor()
    c.execute("""SELECT DISTINCT article_id, ranked
                FROM (
                        SELECT article_id, index_group_name, day_type,
                        RANK() OVER (PARTITION BY index_group_name,day_type ORDER BY total_purchases DESC) AS ranked
                        FROM popular) R
                ORDER BY ranked
                LIMIT 5 OFFSET ?""", (offset,))
    products = c.fetchall()
    c.close()
    return products

@st.cache_resource
def get_product_details(product_id):
    conn = sqlite3.connect('data//mini_product_card.db')
    c = conn.cursor()
    c.execute("""SELECT prod_name, detail_desc
        FROM name_desc_id
        WHERE article_id = ?;""", (product_id,))
    products = c.fetchall()
    c.close()
    return products

@st.cache_resource
def get_description_features():
    conn = sqlite3.connect('data//product_description.db')
    c = conn.cursor()
    c.execute("""SELECT *
        FROM detail_desc
        """)
    products = c.fetchall()
    c.close()
    return pd.DataFrame(products).drop(columns=[0], axis=1)

@st.cache_resource
def get_product_desc(description):
    conn = sqlite3.connect('data//mini_product_card.db')
    c = conn.cursor()
    c.execute("""SELECT article_id, prod_name, detail_desc 
              FROM name_desc_id 
              WHERE detail_desc IN ({})
              """.format(','.join('?' * 5)), description)
    products = c.fetchall()
    c.close()
    return products

def detail_process(user_input, model_nlp):
    """
    Preprocess user input and reduce feature size
    """
    pca_model_path = 'assets'
    pca_model_name = 'good_small_pca_model.pkl'

    # Locating the pca model location
    pca_loc = os.path.join(pca_model_path, pca_model_name)
    
    #Get embeddings for the sentences
    sentence_embeddings = model_nlp.encode(user_input).reshape(1, -1)
    
    # reduce number of features:
    with open(pca_loc, 'rb') as f:
        pca = pickle.load(f)
    processed = pca.transform(sentence_embeddings)
    
    return processed

@st.cache_resource
def find_neighbors(df):
    # NearestNeighbors model(taking 6 since first item will be the item itself)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(df.drop(columns=146, axis=1))
    return nbrs

@st.cache_resource
def get_images(article_id):
    file_name = '0' + str(article_id) +'.jpg'
    img_path = os.path.join('Processed_Images_2', file_name)
    return img_path

@st.cache_resource
def get_full_df():
    conn = sqlite3.connect('data//mini_production_processed.db')
    c = conn.cursor()
    c.execute("""SELECT *
        FROM main_processed
        """)
    products = c.fetchall()
    c.close()
    return pd.DataFrame(products)

# First Section:
st.subheader("Search any item:")
st.write("Describe the the type of product you are looking for and we will recommend the closest match!")

user_text = st.text_input('Product Search')

if user_text == 'pant':
    user_text = 'pants'

if user_text:
    st.session_state['show_search_items'] = True
    st.session_state['user_input'] = user_text

if st.session_state['show_search_items']:
    #st.session_state['show_search_items'] = True

    # All descriptions:
    X = get_description_features()
    user_input = detail_process(st.session_state['user_input'], nlp_model)
    Neighbors = find_neighbors(X)

     # get distances and indices of nearest neighbors
    _, indices = Neighbors.kneighbors(user_input)
    
    similar_descriptions = list(X.iloc[indices[0],145])
    st.session_state['similar_descriptions'] = similar_descriptions

    # Display the 5 similar items:
    products = get_product_desc(st.session_state['similar_descriptions'])
    
    if st.session_state['show_search_items']:
        search_results = st.empty()
        with search_results.container():
            st.text('Here are the closest matches:')
            col_search = st.columns(len(products), gap='medium')
            for i in range(len(products)):
                with col_search[i]:
                    st.session_state['search_article_id'] = products[i][0]
                    a = get_product_details(st.session_state['search_article_id'])

                    st.image(get_images(st.session_state['search_article_id']))

                    st.subheader(a[0][0])

                    st.write(a[0][1])

                    if st.button('Show more like this!', key='item_chosen_'+str(products[i][0])):
                        st.session_state['num_items'] = 0
                        st.session_state['article_liked'] = st.session_state['search_article_id']
                        st.session_state['show_rec'] = True
            # if st.button('Hide Search Result'):
            #     st.session_state['show_search_items'] = False
            #     st.session_state['num_items'] = 0
            #     st.session_state['user_input'] = ""
            #     search_results.empty()


    # st.write('---')

#if st.session_state['show_rec']:

def show_recommended_items(item_liked):
    if st.session_state['show_rec']:
        st.session_state['rec_article_id'] = st.session_state['article_liked']
        items_you_may_like = st.empty()
        
        full_df = get_full_df()

        #st.table(full_df)
        #st.write(type(item_liked))
        item_id = similar_items(full_df, item_liked)
        dim = len(item_id)
        with items_you_may_like.container():
            st.subheader('Items you may like:')
            # rec_results = st.empty()
            st.text('Here are some more items you may like:')
            col_recommended = st.columns(dim, gap='medium')
            for i in range(dim):
                with col_recommended[i]:
                    st.session_state['rec_article_id'] = item_id[i]
                    a = get_product_details(st.session_state['rec_article_id'])
                    
                    st.image(get_images(st.session_state['rec_article_id']))

                    st.subheader(a[0][0])

                    st.write(a[0][1])

                    if st.button('View More', key='item_'+str(item_id[i])):
                        st.session_state['num_items'] = 0
                       # st.session_state['rec_article_id'] = item_liked
                        st.session_state['article_liked'] = st.session_state['rec_article_id']
                        st.session_state['show_search_items'] = False
                        item_liked = st.session_state['rec_article_id']
                       # st.experimental_rerun()
    st.write('---')


def show_popular_items(offset):
    st.subheader('Items sorted by popularity:')
    products = get_products(offset)
    col = st.columns(5, gap='medium')
    for i in range(5):
            with col[i]:
                st.session_state['popular_article_id'] = products[i][0]
                a = get_product_details(st.session_state['popular_article_id'])

                st.image(get_images(st.session_state['popular_article_id']))

                st.subheader(a[0][0])

                st.write(a[0][1])

                if st.button('Click me!', key='main_' + str(products[i][0])):
                    st.session_state['article_liked'] = st.session_state['popular_article_id']
                    st.session_state['show_rec'] = True
                    st.experimental_rerun()
                    #show_recommended_items(st.session_state['rec_article_id'])
                    #st.session_state['show_popular_items'] = False
    if st.button('Show more', key='show_more'):
        st.session_state['num_items'] += 5
st.write('---')


def app():
    show_recommended_items(st.session_state['article_liked'])

    show_popular_items(st.session_state['num_items'])

#show_items(st.session_state['num_items'])


if __name__ == '__main__':
    app()
