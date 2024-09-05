#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests


# In[ ]:





# In[2]:


import requests

def print_api_data(api_url):
    try:
        # Make a GET request to the API
        response = requests.get(api_url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Decode the JSON response
            api_data = response.json()

            # Print the data (assuming it's a JSON response)
            print(api_data)
        else:
            # If the request was not successful, print the error message
            print(f"Error: Unable to fetch data from API. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        # Handle any network-related issues
        print(f"Error: {e}")

# Example API URL
api_url = "https://erp.triz.co.in/lms/o-net-data-category/show?id=3&category-name=knowledge&type=API"


# Call the function to print data from the API
print_api_data(api_url)


# In[3]:


import requests

def fetch_api_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()  # Parse JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None



# Fetch data from API
api_data = fetch_api_data(api_url)

if api_data:
    
    df = pd.DataFrame(api_data)
    
    
    print("DataFrame created successfully:")
    print(df.head())  # Display the first few rows of the DataFrame
else:
    print("Failed to fetch data from the API.")


# In[4]:


sub_categories_data = df['data'][0]['sub_categories']

# Initialize lists to store extracted data
ids = []
category_ids = []
sub_category_names = []
descriptions = []
created_at_list = []
updated_at_list = []
child_ids = []
is_childs_list = []
is_sub_childs_list = []
is_parent_sub_child_list = []

# Iterate through each sub-category and extract relevant fields
for sub_category in sub_categories_data:
    ids.append(sub_category['id'])
    category_ids.append(sub_category['o_net_data_category_id'])
    sub_category_names.append(sub_category['sub_category_name'])
    descriptions.append(sub_category['description'])
    created_at_list.append(sub_category['created_at'])
    updated_at_list.append(sub_category['updated_at'])
    child_ids.append(sub_category['child_id'])
    is_childs_list.append(sub_category['is_childs'])
    is_sub_childs_list.append(sub_category['is_sub_childs'])
    is_parent_sub_child_list.append(sub_category['is_parent_sub_child'])

# Create a DataFrame from the extracted data
df = pd.DataFrame({
    'id': ids,
    'o_net_data_category_id': category_ids,
    'sub_category_name': sub_category_names,
    'description': descriptions,
    'created_at': created_at_list,
    'updated_at': updated_at_list,
    'child_id': child_ids,
    'is_childs': is_childs_list,
    'is_sub_childs': is_sub_childs_list,
    'is_parent_sub_child': is_parent_sub_child_list
})

# Display the DataFrame
print(df)


# In[5]:


df


# In[ ]:





# In[6]:


def print_api_data(api_url1):
    try:
        # Make a GET request to the API
        response = requests.get(api_url1)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Decode the JSON response
            api_data = response.json()

            # Print the data (assuming it's a JSON response)
            print(api_data)
        else:
            # If the request was not successful, print the error message
            print(f"Error: Unable to fetch data from API. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        # Handle any network-related issues
        print(f"Error: {e}")

# Example API URL
api_url1 = "https://erp.triz.co.in/lms_data?table=lms_lesson_plan&sub_institute_id=1"



# Call the function to print data from the API
print_api_data(api_url1)


# In[7]:


import requests

def fetch_api_data(api_url1):
    try:
        response = requests.get(api_url1)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()  # Parse JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None



# Fetch data from API
api_data1 = fetch_api_data(api_url1)

if api_data1:
    
    df1 = pd.DataFrame(api_data1)
    
    
    print("DataFrame created successfully:")
    print(df1.head())  # Display the first few rows of the DataFrame
else:
    print("Failed to fetch data from the API.")


# In[8]:


df1


# In[9]:


df1.info()


# In[10]:


df1.head()


# In[11]:


get_ipython().system('pip install spacy')
import spacy
spacy.cli.download("en_core_web_sm")



# In[12]:


df.info()


# In[16]:


df


# In[ ]:





# In[ ]:





# In[13]:


import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")


# Function to preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Preprocess text 
df["Description_Processed"] = df["description"].apply(preprocess)

# Combine processed description into single string
df["Combined"] = df["sub_category_name"] + " " + df["Description_Processed"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the data
tfidf_matrix1 = vectorizer.fit_transform(df1["subject_id"].astype(str))  # Converting to string for vectorization
tfidf_matrix2 = vectorizer.transform(df["Combined"])

# Calculate cosine similarity between the two matrices
cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Find the most similar row in df2 for each row in df1
matches = []
for i, row in df1.iterrows():
    similar_index = cosine_similarities[i].argmax()
    match = (row["subject_id"], df.iloc[similar_index]["o_net_data_category_id"])
    matches.append(match)

# Create a DataFrame from the matches
matches_df = pd.DataFrame(matches, columns=["subject_id", "o_net_data_category_id"])
print("Matches:")
print(matches_df)


# In[25]:


import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)



# Preprocess text in the relevant columns
df1["Processed_Column"] = df1["learningknowledge"].apply(preprocess)
df["Processed_Column"] = df["Description_Processed"].apply(preprocess)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix1 = vectorizer.fit_transform(df1["Processed_Column"])
tfidf_matrix2 = vectorizer.transform(df["Processed_Column"])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Find the most similar rows
matches = []
for i, row in df1.iterrows():
    similar_index = cosine_similarities[i].argmax()
    match = (row["learningknowledge"], df.iloc[similar_index]["Description_Processed"])
    matches.append(match)

# Create a DataFrame from the matches
matches_df = pd.DataFrame(matches, columns=["learningknowledge", "Description"])
print(matches_df)


# In[15]:


import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)



# Preprocess text in the relevant columns
df1["Processed_Column"] = df1["learningknowledge"].apply(preprocess)
df["Processed_Column"] = df["description"].apply(preprocess)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix1 = vectorizer.fit_transform(df1["Processed_Column"])
tfidf_matrix2 = vectorizer.transform(df["Processed_Column"])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Find the most similar rows
matches = []
for i, row in df1.iterrows():
    similar_index = cosine_similarities[i].argmax()
    match = (row["learningknowledge"], df.iloc[similar_index]["description"])
    matches.append(match)

# Create a DataFrame from the matches
matches_df = pd.DataFrame(matches, columns=["learningknowledge", "description"])
print(matches_df)


# In[26]:


import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)



# Preprocess text in the relevant columns
df1["Processed_Column"] = df1["learningknowledge"].apply(preprocess)
df["Processed_Column"] = df["description"].apply(preprocess)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix1 = vectorizer.fit_transform(df1["Processed_Column"])
tfidf_matrix2 = vectorizer.transform(df["Processed_Column"])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

# Find the most similar rows
matches = []
for i, row in df1.iterrows():
    similar_index = cosine_similarities[i].argmax()
    match = (row["learningknowledge"], df.iloc[similar_index]["sub_category_name"])
    matches.append(match)

# Create a DataFrame from the matches
matches_df = pd.DataFrame(matches, columns=["learningknowledge", "sub_category_name"])
print(matches_df)


# In[27]:


print(matches_df["learningknowledge"].values)


# In[28]:


matches_df


# In[ ]:




