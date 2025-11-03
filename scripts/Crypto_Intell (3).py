#!/usr/bin/env python
# coding: utf-8

# # Scrape Hyperlinks below

# In[1]:


import requests
from bs4 import BeautifulSoup
URL = ['https://rekt.news/leaderboard/']
links = []
for i in URL:
# get page content with url using requests.get()
    page = requests.get(i)
    # get page contnet with BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')
    for a_href in soup.find_all("a", href=True):
          if len(a_href["href"]) > 1 and a_href["href"].startswith("/"):
                links.append('https://rekt.news'+a_href["href"])
links.pop(0)
links.pop(0)
#print(links)


# In[2]:


def get_text(url):

    # url = 'https://rekt.news/ronin-rekt/'
    # get page content with url using requests.get()
    page = requests.get(url)
    # get page contnet with BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')
    post = soup.find("section", "post-content")
    text = ""
    title = url.split("/")[-2]
    for p in post.find_all('p')[:-5]:
        text += p.text
    return title, text


# In[3]:


def get_text(url):
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Get the page content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get title - try multiple methods
        title = url.split("/")[-2]  # default to URL segment
        title_tag = soup.find("h1", {"class": "post-title"})
        if title_tag:
            title = title_tag.text.strip()
        
        # Initialize text
        text = ""
        
        # Try multiple selectors for content
        post = (
            soup.find("section", {"class": "post-content"}) or
            soup.find("article") or
            soup.find("main") or
            soup.find("div", {"class": "content"})
        )
        
        if post:
            # Get all paragraphs
            paragraphs = post.find_all('p')
            
            # Process paragraphs
            for p in paragraphs:
                # Skip if it's a sharing section or end marker
                if any(skip in p.text.lower() for skip in ['share this article', 'follow us']):
                    continue
                    
                # Skip if it contains the end marker image
                if p.find('img') and 'rekt-outline-conc.png' in p.find('img').get('src', ''):
                    continue
                    
                text += " " + p.text.strip()
            
            text = text.strip()
            
            if text:  # Only return if we found some text
                return title, text
            else:
                print(f"[W] No text content found in {url}")
                return None, None
        else:
            print(f"[W] Could not find content section in {url}")
            return None, None
            
    except Exception as e:
        print(f"[E] Error processing {url}: {str(e)}")
        return None, None

# Main scraping code
def scrape_links(links):
    data = dict()
    total = len(links)
    
    for idx, link in enumerate(links, 1):
        print(f"[{idx}/{total}] Processing: {link}")
        
        if not link:  # Skip empty links
            continue
            
        res = get_text(link)
        
        if res[0] and res[1]:  # Only add if we got valid data
            data[res[0]] = res[1]
            print(f"✅ Successfully scraped: {res[0][:50]}...")
        else:
            print(f"❌ Failed to scrape: {link}")
        
        print("-" * 50)
    
    print(f"\nCompleted scraping {len(data)} out of {total} links")
    return data

# Usage:
# First make sure links is defined
# links = ["url1", "url2", ...]
data = scrape_links(links)


# In[4]:


"""
Crypto_Intell (2).ipynb"#scrap text from all links 
data = dict()
for link in links:
    res = get_text(link)

    data[res[0]] = res[1]
    #print(data)
"""


# # Caculate Word Severity Index 

# In[5]:


from pprint import pprint
import pickle
import re
import requests
import urllib.request
import urllib.parse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.preprocessing import minmax_scale
#import credentials as credentials_csv


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer  # default lemmatizer
from nltk import sent_tokenize, word_tokenize


# Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


with open('NRC-VAD-Lexicon.txt') as file:
    data_list = []
    line = file.readline()
    while line:
        data_list.append(str(line))
        line = file.readline()


# In[7]:


# Split and clean
data_list2 = [x.replace('\n', '').split('\t') for x in data_list]


# In[8]:


data_list2[:5]


# In[9]:


#vad_lexicon = pd.DataFrame(data_list2[1:], columns=data_list2[0], dtype=float)
#vad_lexicon.columns = (col.lower() for col in vad_lexicon.columns)


# First create DataFrame without forcing float conversion
vad_lexicon = pd.DataFrame(data_list2[1:], columns=data_list2[0])

# Convert column names to lowercase
vad_lexicon.columns = [col.lower() for col in vad_lexicon.columns]

# Now let's see what columns we have
print("Columns:", vad_lexicon.columns.tolist())

# Assuming the first column might be words/text and the rest should be numeric
text_column = vad_lexicon.columns[0]  # Usually the first column is text
numeric_columns = vad_lexicon.columns[1:]  # Rest should be numeric

# Convert only numeric columns to float
for col in numeric_columns:
    vad_lexicon[col] = pd.to_numeric(vad_lexicon[col], errors='coerce')

# Print info about the DataFrame
print("\nDataFrame Info:")
print(vad_lexicon.info())


# In[10]:


# Check results ...
display(vad_lexicon.iloc[[15201]])


# ## Understanding the Code: Emotional Impact Analysis of Crypto Terms

# 
# This code analyses emotional impact scores for cryptocurrency-related terms using the NRC VAD (Valence-Arousal-Dominance) Lexicon. Here's what it does:
# 
# Part 1: Data Loading and WBI Calculation
# Loads the NRC VAD Lexicon from 'NRC-VAD-Lexicon.txt' as a CSV file using commas as separators.
# Standardises column names to lowercase for consistency.
# Calculates the WBI (Word Brutality Index) by:
# Computing anti_valence = 1 - valence (inverting the emotional tone)
# Calculating WBI = minmax_scale(anti_valence + arousal) (normalising the combined score)
# Cleans up by removing the original 'valence' and 'dominance' columns
# Sorts terms by WBI score in descending order
# 
# Part 2: Categorisation and Analysis
# The code organises terms into three main categories:
# 
# Major Hacks - Grouped by financial impact:
# Tier 1: Over $300 million (e.g., Ronin, Poly, Wormhole)
# Tier 2: $100-300 million (e.g., FTX, Nomad)
# Tier 3: $50-100 million (e.g., Compound, Venus)
# Tier 4: $20-50 million (e.g., PancakeBunny, Uranium)
# Vulnerabilities - Technical weaknesses including:
# Access Control issues (e.g., reentrancy, unauthorised access)
# Logic/Arithmetic errors (e.g., overflow, underflow)
# DeFi-specific risks (e.g., flash loans, oracle manipulation)
# And other technical categories
# Crypto Terminology - General blockchain terms covering:
# Infrastructure (blockchain, smart contracts)
# DeFi mechanisms (staking, liquidity pools)
# Wallets and transactions
# And other relevant terms
# Part 3: Analysis and Output
# The print_category_wbi function:
# 
# Iterates through each category and subcategory
# Finds matching terms in the lexicon
# Calculates and displays:
# Each term's WBI score
# Average WBI per subcategory
# Shows both overall results and highlights high-risk terms (top 25% WBI scores)
# Key Outputs:
# WBI scores for all categorised terms
# Average WBI per subcategory
# Identification of high-risk terms
# Sorted lists showing the most "brutal" terms in each category
# Purpose:
# This analysis helps identify which cryptocurrency-related terms have the strongest negative emotional impact (high WBI), which is useful for:
# 
# Risk assessment
# Sentiment analysis
# Understanding emotional triggers in crypto discussions
# Identifying particularly concerning vulnerabilities based on their emotional impact
# The WBI metric combines how unpleasant (anti-valence) and emotionally intense (arousal) each term is, with scores normalised for comparison across the entire lexicon.

# In[11]:


# Read the CSV file with the correct separator
vad_lexicon = pd.read_csv('NRC-VAD-Lexicon.txt', sep=',')
print("Original columns:", vad_lexicon.columns.tolist())

# Make a copy and convert column names to lowercase
lexicon = vad_lexicon.copy()
lexicon.columns = lexicon.columns.str.lower()
print("Columns after lowercase:", lexicon.columns.tolist())

# Now proceed with the calculations
lexicon['anti_valence'] = lexicon['valence'].apply(lambda x: 1-x)
wbi = minmax_scale(lexicon['anti_valence'] + lexicon['arousal'])
lexicon['wbi'] = wbi
lexicon.drop(['valence', 'dominance'], axis=1, inplace=True)
lexicon = lexicon.sort_values(['wbi'], ascending=False).reset_index(drop=True)

# Print final result
print("\nFinal columns:", lexicon.columns.tolist())
print("\nTop 10 terms by WBI:")
print(lexicon.head(10))


import pandas as pd
from sklearn.preprocessing import minmax_scale
import numpy as np

# Read and process the lexicon
vad_lexicon = pd.read_csv('NRC-VAD-Lexicon.txt', sep=',')
lexicon = vad_lexicon.copy()
lexicon.columns = lexicon.columns.str.lower()
lexicon['anti_valence'] = lexicon['valence'].apply(lambda x: 1-x)
wbi = minmax_scale(lexicon['anti_valence'] + lexicon['arousal'])
lexicon['wbi'] = wbi
lexicon.drop(['valence', 'dominance'], axis=1, inplace=True)

# Define our categories
categories = {
    'Major Hacks': {
        'Tier 1 (>$300M)': ['ronin', 'poly', 'wormhole'],
        'Tier 2 ($100M-$300M)': ['ftx', 'nomad', 'beanstalk', 'wintermute', 'cream', 'badger', 'harmony'],
        'Tier 3 ($50M-$100M)': ['compound', 'venus', 'qubit'],
        'Tier 4 ($20M-$50M)': ['pancakebunny', 'uranium', 'vulcan', 'oneringrekt']
    },
    'Vulnerabilities': {
        'Access Control': ['reentrancy', 'accesscontrol', 'unauthorized', 'privilegeescalation'],
        'Logic/Arithmetic': ['overflow', 'underflow', 'integeroverflow', 'roundingerror'],
        'DeFi-Specific': ['flashloanattack', 'oraclemanipulation', 'pricemanipulation', 'arbitrage'],
        'Implementation': ['racecondition', 'frontrunning', 'timestampmanipulation', 'randomness'],
        'External Interaction': ['crosscontract', 'bridgeexploit', 'callinjection', 'delegatecall'],
        'Storage/State': ['storagecollision', 'uninitializedstate', 'statemanipulation'],
        'Gas-Related': ['gasgriefing', 'doslimit', 'gasmanipulation']
    },
    'Crypto Terminology': {
        'Infrastructure': ['blockchain', 'defi', 'cefi', 'smart_contract', 'dao', 'dex', 'amm', 'layer1', 'layer2', 'sidechain'],
        'DeFi Mechanisms': ['yield', 'staking', 'farming', 'liquidity', 'pool', 'swap', 'bridge', 'mint', 'burn'],
        'Wallet/Transaction': ['wallet', 'coldwallet', 'hotwallet', 'multisig', 'gas', 'gwei', 'nonce'],
        'Token Standards': ['erc20', 'erc721', 'erc1155', 'fungible', 'nonfungible'],
        'NFT/Gaming': ['nft', 'gamefi', 'metaverse', 'play2earn'],
        'Trading': ['bullish', 'bearish', 'hodl', 'fomo', 'fud', 'dyor', 'ath', 'atl'],
        'Consensus/Mining': ['pow', 'pos', 'mining', 'validator', 'node', 'hashrate'],
        'Governance': ['governance', 'proposal', 'vote', 'quorum', 'snapshot'],
        'Technical Analysis': ['resistance', 'support', 'breakout', 'breakdown', 'consolidation'],
        'Development': ['fork', 'hardfork', 'softfork', 'testnet', 'mainnet', 'deployment'],
        'Community': ['whitepaper', 'roadmap', 'airdrop', 'whitelist', 'community', 'tokenomics']
    }
}

def print_category_wbi(lexicon_df, categories, min_wbi=None):
    """Print WBI scores for each category, sorted by WBI score"""
    print("\n=== WBI Scores by Category ===")
    
    for main_category, subcategories in categories.items():
        print(f"\n\n{main_category.upper()}")
        print("=" * len(main_category))
        
        for subcategory, terms in subcategories.items():
            print(f"\n{subcategory}:")
            print("-" * len(subcategory))
            
            # Get WBI scores for terms in this category
            category_scores = []
            for term in terms:
                term_data = lexicon_df[lexicon_df['word'].str.lower() == term.lower()]
                if not term_data.empty:
                    wbi_score = term_data['wbi'].iloc[0]
                    if min_wbi is None or wbi_score >= min_wbi:
                        category_scores.append((term, wbi_score))
            
            # Sort by WBI score and print
            if category_scores:
                category_scores.sort(key=lambda x: x[1], reverse=True)
                for term, score in category_scores:
                    print(f"{term:<20} WBI: {score:.3f}")
                print(f"Average WBI: {sum(score for _, score in category_scores) / len(category_scores):.3f}")
            else:
                print("No terms found in lexicon")

# Print all categories with their WBI scores
print_category_wbi(lexicon, categories)

# Optional: Print only high WBI scores (e.g., top 25%)
wbi_threshold = np.percentile(lexicon['wbi'], 75)
print("\n\nHIGH RISK TERMS (Top 25% WBI Scores):")
print(f"WBI Threshold: {wbi_threshold:.3f}")
print_category_wbi(lexicon, categories, min_wbi=wbi_threshold)


# In[12]:


# Check results ...
display(lexicon.nlargest(10, 'wbi'))
display(lexicon.loc[lexicon['word'] == 'zombie'])


# Define the terms to check
terms_to_check = {
    'MAJOR HACKS': [
        'ronin', 'poly', 'wormhole',  # Tier 1
        'ftx', 'nomad', 'beanstalk', 'wintermute', 'cream', 'badger', 'harmony',  # Tier 2
        'compound', 'venus', 'qubit',  # Tier 3
        'pancakebunny', 'uranium', 'vulcan', 'oneringrekt'  # Tier 4
    ],
    'VULNERABILITIES': [
        'reentrancy', 'accesscontrol', 'unauthorized', 'privilegeescalation',
        'overflow', 'underflow', 'integeroverflow', 'roundingerror',
        'flashloanattack', 'oraclemanipulation', 'pricemanipulation', 'arbitrage',
        'racecondition', 'frontrunning', 'timestampmanipulation', 'randomness',
        'crosscontract', 'bridgeexploit', 'callinjection', 'delegatecall',
        'storagecollision', 'uninitializedstate', 'statemanipulation',
        'gasgriefing', 'doslimit', 'gasmanipulation'
    ],
    'CRYPTO TERMINOLOGY': [
        'blockchain', 'defi', 'cefi', 'smart_contract', 'dao', 'dex', 'amm',
        'yield', 'staking', 'farming', 'liquidity', 'pool', 'swap', 'bridge',
        'wallet', 'coldwallet', 'hotwallet', 'multisig', 'gas', 'nonce',
        'erc20', 'erc721', 'erc1155', 'nft', 'gamefi', 'metaverse',
        'bullish', 'bearish', 'hodl', 'fomo', 'fud', 'dyor',
        'pow', 'pos', 'mining', 'validator', 'node', 'hashrate',
        'governance', 'proposal', 'vote', 'quorum', 'snapshot'
    ]
}

# Print results for each category
for category, terms in terms_to_check.items():
    print(f"\n=== {category} ===")
    print("-" * (len(category) + 8))
    
    # Get WBI scores for all terms in this category
    category_results = []
    for term in terms:
        term_data = lexicon.loc[lexicon['word'].str.lower() == term.lower()]
        if not term_data.empty:
            category_results.append(term_data.iloc[0])
    
    if category_results:
        # Convert to DataFrame and sort by WBI
        category_df = pd.DataFrame(category_results)
        category_df = category_df.sort_values('wbi', ascending=False)
        
        # Display results
        print(f"\nTop terms by WBI in {category}:")
        display(category_df)
        
        # Show statistics
        print(f"\nStatistics for {category}:")
        print(f"Average WBI: {category_df['wbi'].mean():.3f}")
        print(f"Max WBI: {category_df['wbi'].max():.3f}")
        print(f"Min WBI: {category_df['wbi'].min():.3f}")
    else:
        print(f"No terms found for {category}")

# Show overall top 10 for comparison
print("\n=== OVERALL TOP 10 TERMS BY WBI ===")
print("-" * 30)
display(lexicon.nlargest(10, 'wbi'))


# In[13]:


# Save a copy of the lexicon
lexicon.to_csv('wbi_lexicon_cc.csv', index=False)


# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('wbi_lexicon_cc.csv')
# Add this at the beginning of your code, right after loading the data
df['word'] = df['word'].astype(str)

# Then modify the get_category function to be more robust
def get_category(word, categories):
    try:
        word_lower = str(word).lower().strip()
        for main_cat, subcats in categories.items():
            for subcat, terms in subcats.items():
                if word_lower in [t.lower() for t in terms]:
                    return main_cat, subcat
    except:
        pass
    return None, None

# Then continue with the rest of your code...

# Define the categories
categories = {
    'Major Hacks': {
        'Tier 1 (>$300M)': ['ronin', 'poly', 'wormhole'],
        'Tier 2 ($100M-$300M)': ['ftx', 'nomad', 'beanstalk', 'wintermute', 'cream', 'badger', 'harmony'],
        'Tier 3 ($50M-$100M)': ['compound', 'venus', 'qubit'],
        'Tier 4 ($20M-$50M)': ['pancakebunny', 'uranium', 'vulcan', 'oneringrekt']
    },
    'Vulnerabilities': {
        'Access Control': ['reentrancy', 'accesscontrol', 'unauthorized', 'privilegeescalation'],
        'Logic/Arithmetic': ['overflow', 'underflow', 'integeroverflow', 'roundingerror'],
        'DeFi-Specific': ['flashloanattack', 'oraclemanipulation', 'pricemanipulation', 'arbitrage'],
        'Implementation': ['racecondition', 'frontrunning', 'timestampmanipulation', 'randomness'],
        'External Interaction': ['crosscontract', 'bridgeexploit', 'callinjection', 'delegatecall'],
        'Storage/State': ['storagecollision', 'uninitializedstate', 'statemanipulation'],
        'Gas-Related': ['gasgriefing', 'doslimit', 'gasmanipulation']
    },
    'Crypto Terminology': {
        'Infrastructure': ['blockchain', 'defi', 'cefi', 'smart_contract', 'dao', 'dex', 'amm', 'layer1', 'layer2', 'sidechain'],
        'DeFi Mechanisms': ['yield', 'staking', 'farming', 'liquidity', 'pool', 'swap', 'bridge', 'mint', 'burn'],
        'Wallet/Transaction': ['wallet', 'coldwallet', 'hotwallet', 'multisig', 'gas', 'gwei', 'nonce'],
        'Token Standards': ['erc20', 'erc721', 'erc1155', 'fungible', 'nonfungible'],
        'NFT/Gaming': ['nft', 'gamefi', 'metaverse', 'play2earn'],
        'Trading': ['bullish', 'bearish', 'hodl', 'fomo', 'fud', 'dyor', 'ath', 'atl'],
        'Consensus/Mining': ['pow', 'pos', 'mining', 'validator', 'node', 'hashrate'],
        'Governance': ['governance', 'proposal', 'vote', 'quorum', 'snapshot'],
        'Technical Analysis': ['resistance', 'support', 'breakout', 'breakdown', 'consolidation'],
        'Development': ['fork', 'hardfork', 'softfork', 'testnet', 'mainnet', 'deployment'],
        'Community': ['whitepaper', 'roadmap', 'airdrop', 'whitelist', 'community', 'tokenomics']
    }
}

# Function to get category for a word
def get_category(word, categories):
    word_lower = word.lower()
    for main_cat, subcats in categories.items():
        for subcat, terms in subcats.items():
            if word_lower in terms:
                return main_cat, subcat
    return None, None

# Add category information to the dataframe
df[['main_category', 'subcategory']] = df['word'].apply(
    lambda x: pd.Series(get_category(x, categories))
)

# Calculate percentiles for overall distribution
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
wbi_percentiles = np.percentile(df['wbi'].dropna(), percentiles)

# Create distribution table
dist_table = pd.DataFrame({
    'Percentile': [f'{p}%' for p in percentiles],
    'WBI_Score': wbi_percentiles
})

# Display the distribution table
print("WBI Score Distribution:")
display(dist_table)

# Basic statistics
print("\nBasic Statistics:")
print(f"Mean WBI: {df['wbi'].mean():.4f}")
print(f"Median WBI: {df['wbi'].median():.4f}")
print(f"Standard Deviation: {df['wbi'].std():.4f}")
print(f"Minimum WBI: {df['wbi'].min():.4f}")
print(f"Maximum WBI: {df['wbi'].max():.4f}")

# Plot the distribution with categories
plt.figure(figsize=(14, 7))

# Plot overall distribution
sns.histplot(df, x='wbi', bins=30, kde=True, color='lightgray', 
             label='All Terms', alpha=0.5)

# Plot each main category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (cat, color) in enumerate(zip(categories.keys(), colors)):
    cat_data = df[df['main_category'] == cat]
    if not cat_data.empty:
        sns.histplot(cat_data, x='wbi', bins=30, color=color, 
                    alpha=0.5, label=f'{cat} (n={len(cat_data)})')

# Add percentile lines
for i, p in enumerate([25, 50, 75, 90, 95]):
    plt.axvline(x=wbi_percentiles[percentiles.index(p)], 
                color='red' if i % 2 == 0 else 'blue', 
                linestyle='--', 
                alpha=0.7,
                label=f'{p}th: {wbi_percentiles[percentiles.index(p)]:.2f}')

plt.title('Distribution of WBI Scores by Category')
plt.xlabel('WBI Score')
plt.ylabel('Frequency')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Category-wise analysis
print("\n" + "="*50)
print("CATEGORY-WISE ANALYSIS")
print("="*50)

for cat in categories.keys():
    cat_data = df[df['main_category'] == cat]
    if not cat_data.empty:
        print(f"\n{cat.upper()} (n={len(cat_data)})")
        print("-" * (len(cat) + 5))
        
        # Show top terms
        top_terms = cat_data.nlargest(5, 'wbi')[['word', 'wbi']]
        print("\nTop 5 Terms by WBI:")
        display(top_terms)
        
        # Show statistics
        print("\nStatistics:")
        stats = cat_data['wbi'].describe(percentiles=[.25, .5, .75, .9, .95])
        display(stats.to_frame('WBI Score'))
        
        # Subcategory analysis
        print("\nSubcategory Analysis:")
        subcat_stats = cat_data.groupby('subcategory')['wbi'].agg(['count', 'mean', 'min', 'max'])
        display(subcat_stats.sort_values('mean', ascending=False))
        
        # Plot subcategory distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=cat_data, x='subcategory', y='wbi')
        plt.title(f'WBI Distribution by {cat} Subcategory')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Show top and bottom 5 terms overall
print("\n" + "="*50)
print("TOP AND BOTTOM TERMS OVERALL")
print("="*50)

print("\nTop 10 Terms by WBI:")
display(df.nlargest(10, 'wbi')[['word', 'wbi', 'main_category', 'subcategory']])

print("\nBottom 10 Terms by WBI:")
display(df.nsmallest(10, 'wbi')[['word', 'wbi', 'main_category', 'subcategory']])

# Save the categorized data
df.to_csv('categorised_wbi_lexicon.csv', index=False)
print("\nCategorised data saved to 'categorised_wbi_lexicon.csv'")


# let's say that the top 10% of WBI scores are the 'brutal' words. This will be the threshold we will use when calculating the "Lyrical Brutality Index" later on. All words with a wbi score higher than that will be considered 'brutal'. The threshold's value is:

# In[15]:


wbi_threshold = np.percentile(lexicon['wbi'], 90)
print("WBI-threshold for 'brutal' words:", round(wbi_threshold, 3))


# In[16]:


# Look at some words around threshold

print("Just above threshold:")
display(lexicon.loc[lexicon['wbi'] >= 0.723].tail())
print("\nJust below threshold:")
display(lexicon.loc[lexicon['wbi'] < 0.723].head())


def analyze_threshold_boundaries(lexicon_df, terms, category_name, window_size=5):
    """Analyze terms around the threshold for a specific category"""
    # Filter lexicon for the specified terms
    category_terms = lexicon_df[lexicon_df['word'].str.lower().isin([t.lower() for t in terms])]
    
    if not category_terms.empty:
        # Calculate median WBI for this category
        median_wbi = category_terms['wbi'].median()
        
        print(f"\n=== {category_name} ===")
        print(f"Median WBI: {median_wbi:.3f}")
        print("-" * 50)
        
        print(f"\nTerms just above median ({window_size} items):")
        above_median = category_terms[category_terms['wbi'] >= median_wbi].sort_values('wbi')
        display(above_median.head(window_size))
        
        print(f"\nTerms just below median ({window_size} items):")
        below_median = category_terms[category_terms['wbi'] < median_wbi].sort_values('wbi', ascending=False)
        display(below_median.head(window_size))
        
        # Show statistics
        print(f"\nCategory Statistics:")
        print(f"Highest WBI: {category_terms['wbi'].max():.3f} ({category_terms.loc[category_terms['wbi'].idxmax(), 'word']})")
        print(f"Lowest WBI: {category_terms['wbi'].min():.3f} ({category_terms.loc[category_terms['wbi'].idxmin(), 'word']})")
        print(f"Average WBI: {category_terms['wbi'].mean():.3f}")
        
    else:
        print(f"\nNo terms found for {category_name}")

# Define categories
categories = {
    'MAJOR HACKS': [
        'ronin', 'poly', 'wormhole',  # Tier 1
        'ftx', 'nomad', 'beanstalk', 'wintermute', 'cream', 'badger', 'harmony',  # Tier 2
        'compound', 'venus', 'qubit',  # Tier 3
        'pancakebunny', 'uranium', 'vulcan', 'oneringrekt'  # Tier 4
    ],
    'VULNERABILITIES': [
        'reentrancy', 'accesscontrol', 'unauthorized', 'privilegeescalation',
        'overflow', 'underflow', 'integeroverflow', 'roundingerror',
        'flashloanattack', 'oraclemanipulation', 'pricemanipulation', 'arbitrage',
        'racecondition', 'frontrunning', 'timestampmanipulation', 'randomness',
        'crosscontract', 'bridgeexploit', 'callinjection', 'delegatecall',
        'storagecollision', 'uninitializedstate', 'statemanipulation',
        'gasgriefing', 'doslimit', 'gasmanipulation'
    ],
    'CRYPTO TERMINOLOGY': [
        'blockchain', 'defi', 'cefi', 'smart_contract', 'dao', 'dex', 'amm',
        'yield', 'staking', 'farming', 'liquidity', 'pool', 'swap', 'bridge',
        'wallet', 'coldwallet', 'hotwallet', 'multisig', 'gas', 'nonce',
        'erc20', 'erc721', 'erc1155', 'nft', 'gamefi', 'metaverse',
        'bullish', 'bearish', 'hodl', 'fomo', 'fud', 'dyor',
        'pow', 'pos', 'mining', 'validator', 'node', 'hashrate',
        'governance', 'proposal', 'vote', 'quorum', 'snapshot'
    ]
}

# Analyze each category
for category_name, terms in categories.items():
    analyze_threshold_boundaries(lexicon, terms, category_name)

# Show overall lexicon threshold analysis for comparison
print("\n=== OVERALL LEXICON THRESHOLD ANALYSIS ===")
print("-" * 50)
overall_median = lexicon['wbi'].median()
print(f"Overall Median WBI: {overall_median:.3f}")

print("\nOverall terms just above median:")
display(lexicon[lexicon['wbi'] >= overall_median].sort_values('wbi').head())

print("\nOverall terms just below median:")
display(lexicon[lexicon['wbi'] < overall_median].sort_values('wbi', ascending=False).head())


# In[17]:


# Create a lookup dictionary for 'word: wbi' - we'll use it later
wbi_lookup = dict(zip(lexicon['word'], lexicon['wbi']))


# In[18]:


# Save / load dict 
with open('text_dict.pkl', 'wb') as f:
        pickle.dump(data, f)
#with open('C:/Users/Administrator/lyrics_dict.pkl', 'rb') as f:
#    lyrics_dict = pickle.load(f)
# print(data)


# In[19]:


def process_text(raw_text):
    """A simple NLP pipeline returning 'lemmed' tokens."""
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", raw_text.lower().strip())
    tokens = word_tokenize(text)
    lemmed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]  # lemm to verbs, not nouns 
    return lemmed_tokens


# In[20]:


"""
# Method 1: Print all titles and first 100 characters of each text
print("=== Method 1: Overview of all entries ===")
for title, text in data_dict_clean.items():
    print(f"\nTitle: {title}")
    print(f"Text preview: {text[:100]}...")
    print("-" * 50)

"""


# In[21]:


data_dict_clean = {title: process_text(text) for title, text in data.items()}
# print one sample
print(data_dict_clean['Ronin Network - REKT'])


# In[22]:


text_dict_wbi = {}
for title, text in data_dict_clean.items():
    text_dict_wbi[title] = [wbi_lookup[word] for word in text if word in wbi_lookup.keys()]


# In[23]:


# print one sample
print(text_dict_wbi['Ronin Network - REKT'])


# In[24]:


# Just for visualization, display as boolean
show = list(text_dict_wbi['Ronin Network - REKT'] )
show = ['BRUTAL' if x > wbi_threshold else 'False' for x in show]
print(show)


# # Define and calculate the "Lyrical Brutality Index

# I was looking for an score that takes into account how many 'brutal' words are used in a song, what the proportion of these is in relation to the total of the words sung (screamed / growled) and relates them to the duration of a song. So that's what I chose in the end:
# 
# 𝐿𝑦𝑟𝑖𝑐𝑎𝑙𝐵𝑟𝑢𝑡𝑎𝑙𝑖𝑡𝑦𝐼𝑛𝑑𝑒𝑥=(brutal wordsnumber of words)∗(number of wordsduration)⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯√
#  
# 𝐿𝑦𝑟𝑖𝑐𝑎𝑙𝐵𝑟𝑢𝑡𝑎𝑙𝑖𝑡𝑦𝐼𝑛𝑑𝑒𝑥=brutal wordsduration⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯√
#  
# I will use top 10% WBI score of 0.723214 

# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def show_wbi_distribution(scores, percentiles=None, title="WBI Score Distribution", figsize=(12, 6)):
    """
    Visualize the distribution of WBI scores with optional percentile markers.
    
    Parameters:
    - scores: List or array of WBI scores
    - percentiles: List of percentiles to mark (e.g., [90, 95, 99])
    - title: Plot title
    - figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Create histogram
    sns.histplot(scores, bins=50, kde=True, color='skyblue', edgecolor='black')
    
    # Add vertical lines for percentiles
    if percentiles:
        for p in percentiles:
            if 0 < p < 100:  # Ensure percentile is between 0 and 100
                value = np.percentile(scores, p)
                plt.axvline(x=value, color='red', linestyle='--', 
                           label=f'{p}th: {value:.2f}')
    
    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel('WBI Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add legend for percentiles
    if percentiles:
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("WBI Score Statistics:")
    print(f"Count: {len(scores):,}")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    print(f"Min: {np.min(scores):.4f}")
    print(f"25%: {np.percentile(scores, 25):.4f}")
    print(f"50% (Median): {np.median(scores):.4f}")
    print(f"75%: {np.percentile(scores, 75):.4f}")
    print(f"90%: {np.percentile(scores, 90):.4f}")
    print(f"95%: {np.percentile(scores, 95):.4f}")
    print(f"99%: {np.percentile(scores, 99):.4f}")
    print(f"Max: {np.max(scores):.4f}")

# Now you can use it with your data
full_corpus = [score for text in text_dict_wbi.values() for score in text]
show_wbi_distribution(full_corpus, percentiles=[90, 95, 99])


# In[26]:


# Visualize the distribution of wbi scores over the whole CC text corpus (ignore graph titel)
full_corpus = [score for text in text_dict_wbi.values() for score in text]
show_wbi_distribution(full_corpus, percentiles=[90])  # approx the threshold value for brutal words


# In[27]:


def calculate_lb_features(text, wbi_treshold=wbi_threshold):
    """Calculate and return features needed to calculate the lbi."""
    total_words = len(text)
    brutal_words = np.sum(text > wbi_threshold)
    brutal_prop = brutal_words / total_words
    
    return total_words, brutal_words, brutal_prop


# In[28]:


#create a Dataframe with the lb features
lb_features_dict = {title: calculate_lb_features(text) for title, text in text_dict_wbi.items()}
print(list(lb_features_dict.items())[:3])


# In[29]:


# Save lb_features_dict to a file
features_file = 'lb_features.pkl'
try:
    with open(features_file, 'wb') as f:
        pickle.dump(lb_features_dict, f)
    print(f"Saved lb_features_dict to {features_file}")
except Exception as e:
    print(f"Error saving lb_features_dict: {e}")


# ## Ethereum Hacks

# In[30]:


['reentrancy',
'blind signing',
'ERC20 token transfer to self token address',
'Frozen ether',
'Use of Outdated Compiler Version',
'Access Control - Smart Contract Initialization',
'Arbitrary Jump with Function Type Variable',
'Assert Violation',
'Authorization through tx.origin',
'Block values as a proxy for time',
'Call Depth Attack',
'Call to Unknown function via fallback()',
'Code With No Effects',
'Cross-Function Race Condition',
'default fallback address attack',
'Delegate call injection',
'Delegate call to Untrusted Callee',
'DoS',
'DoS With Block Gas Limit',
'DoS with Failed Call',
'DoS with unbounded operations',
'DoS with unexpected revert',
'Erroneous constructor name',
'Erroneous visibility',
'Ether Lost in Transfer',
'Ether lost to orphan address',
'Ethereum Gasless Send',
'Floating Pragma',
'Forcibly Sending Ether to a Contract',
'Hash Collisions With Multiple Variable Length Arguments',
'Immutable Bugs',
'Incorrect Constructor Name',
'Incorrect ERC20 implementation',
'Incorrect function state mutability',
'Incorrect Inheritance Order',
'Insufficient Gas Griefing',
'Integer Overflow and Underflow',
'Keeping Secrets',
'Lack of Proper Signature Verification',
'Manipulated balance',
'Message call with hardcoded gas amount',
'Mishandled Exceptions',
'Missing Protection against Signature Replay Attacks',
'Presence of unused variables',
'Race Conditions',
'Reentrancy Race Condition',
'Requirement Violation',
'Right-To-Left-Override control character (U+202E)',
'Shadowing State Variables',
'Short Address Attack',
'Signature Malleability',
'Source Code Unavailable for review',
'Stack Size Limit',
'State Variable Default Visibility',
'Timestamp Dependency',
'Transaction Order Dependence',
'Transaction Ordering Dependency (TOD)',
'Typecasts',
'Typographical Error',
'Unchecked Call Return Value',
'Unchecked Return Values',
'Underpriced opcodes',
'Unencrypted Private Data On-Chain',
'unexpected call return value',
'Unexpected Ether balance',
'Uninitialized Storage Pointer',
'Unpredictable State',
'Unprotected Ether Withdrawal',
'Unprotected SELFDESTRUCT Instruction',
'Unprotected suicide',
'upgradeable contract',
'Usage of continue in do-while',
'Use of Deprecated Solidity Functions',
'Weak Field Modifier',
'Weak Sources of Randomness from Chain Attributes',
'Write to Arbitrary Storage Location',
'Function Default Visibility',
'Time Manipulation',
'Vulnerabilities in virtual machines',
'Sybil Attack',
'Eclipse Attack',
'Eavesdropping Attack',
'Denial of Service Attack',
'BGP Hijack Attack',
'Alien Attack',
'Timejacking',
'Eavesdropping Attack',
'The Ethereum Black Valentines Day Vulnerability',
'Http Input Attack',
'Cross-Domain Phishing Attack',
'Long Range Attack',
'Bribery Attack',
'Race Attack',
'Liveness Denial',
'Censorship',
'Finney Attack',
'Vector76 Attack',
'Alternative Historical Attack',
'51% Attack',
'Grinding Attack',
'Coin Age Accumulation Attack',
'Selfing Mining',
'Block Double Production',
'Cryptographic Attack',
'Private Key Prediction',
'Length Extension Attack',
'Hash collision attack',
'Transaction Replay Attack',
'Transaction Malleability Attack',
'Time-Locked Transaction Attack',
'False Top-Up Attack',
'Rug Pull Attack']


# ## Machine Learning for North Korean Hacking groups

# How the Hacker Group Attribution ML Works :
# The system converts hack descriptions into numerical vectors using TF-IDF, then applies dimensionality reduction and K-means clustering to group similar hacks together without requiring labeled data. Each hack is scored against known hacker groups based on techniques, targets, regions, and tools mentioned in the text, with higher scores given when characteristics match. The clustering results are used to refine these scores, assuming similar hacks in the same cluster likely come from the same group. Hack titles from the leaderboard are matched to the analysed texts to connect entries with their detailed descriptions. The final attribution combines unsupervised machine learning with domain knowledge to provide both a likely hacker group and a confidence score for each hack.

# In[31]:


import os
import re
import pickle
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.preprocessing import minmax_scale
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variable to avoid KMeans memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

def format_cash_value(amount):
    """Format numeric value as currency with $ and commas"""
    try:
        amount = float(amount)
        return f"${amount:,.2f}"
    except:
        return "$0.00"

def extract_cash_amount(text):
    """Extract cash amount from text like '$1,436,173,027 | 2/21/2025'"""
    if not text:
        return 0.0
    
    # Find the first number (before |) and remove non-numeric characters
    amount_text = text.split('|')[0].strip() if '|' in text else text.strip()
    
    # Handle special cases like $1.4B or $600M
    if 'B' in amount_text or 'b' in amount_text:
        amount_text = amount_text.lower().replace('b', '')
        multiplier = 1000000000
    elif 'M' in amount_text or 'm' in amount_text:
        amount_text = amount_text.lower().replace('m', '')
        multiplier = 1000000
    elif 'K' in amount_text or 'k' in amount_text:
        amount_text = amount_text.lower().replace('k', '')
        multiplier = 1000
    else:
        multiplier = 1
    
    # Remove all non-numeric characters except decimal point
    amount_num = re.sub(r'[^\d.]', '', amount_text)
    
    try:
        return float(amount_num) * multiplier if amount_num else 0.0
    except:
        return 0.0

def convert_date_format(date_str):
    """Convert date from MM/DD/YYYY to DD/MM/YYYY format"""
    if not date_str or not isinstance(date_str, str):
        return ""
    
    try:
        # Parse the date as MM/DD/YYYY
        date_obj = datetime.strptime(date_str.strip(), '%m/%d/%Y')
        # Return as DD/MM/YYYY
        return date_obj.strftime('%d/%m/%Y')
    except:
        return date_str  # Return original if parsing fails

def get_hack_type(title, vulnerability_types):
    """Determine hack type from title using vulnerability patterns"""
    if not title:
        return 'Unknown'
    
    title_lower = title.lower()
    
    # Manual overrides for well-known hacks
    if 'ronin' in title_lower:
        return 'Bridge'
    elif 'poly network' in title_lower:
        return 'Bridge'
    elif 'wormhole' in title_lower:
        return 'Bridge'
    elif 'mango' in title_lower:
        return 'DeFi-Specific'
    elif 'platypus' in title_lower:
        return 'Flash Loan'
    elif 'binance' in title_lower or 'bnb' in title_lower:
        return 'Exchange'
    
    # Check for Ethereum-related hacks
    is_ethereum = any(term in title_lower for term in vulnerability_types.get('Ethereum', []))
    
    # Check general categories
    if 'bridge' in title_lower:
        return 'Bridge'
    elif 'flash' in title_lower or 'loan' in title_lower:
        return 'Flash Loan'
    elif 'oracle' in title_lower or 'price' in title_lower:
        return 'Oracle'
    elif is_ethereum:
        return 'Ethereum'
    
    return 'Unknown'

def scrape_rekt_leaderboard():
    """Scrape the rekt.news leaderboard with improved parsing"""
    # Define vulnerability types for classification
    vulnerability_types = {
     'Access Control': [
            'reentrancy', 'access control', 'unauthorized', 'privilege escalation',
            'authorization through tx.origin', 'unprotected ether withdrawal',
            'unprotected selfdestruct instruction', 'unprotected suicide',
            'access control - smart contract initialization', 'delegate call to untrusted callee',
            'delegate call injection', 'function default visibility', 'state variable default visibility',
            'erroneous visibility', 'weak field modifier'
        ],
        
        'Logic/Arithmetic': [
            'integer overflow and underflow', 'overflow', 'underflow', 'integer', 'rounding', 'arithmetic',
            'typographical error', 'incorrect inheritance order', 'immutable bugs',
            'unexpected call return value', 'unexpected ether balance', 'assert violation',
            'requirement violation', 'code with no effects', 'incorrect constructor name',
            'erroneous constructor name'
        ],
        
        'DeFi-Specific': [
            'flashloan', 'flash loan', 'oracle', 'price manipulation', 'arbitrage',
            'rug pull attack', 'manipulated balance', 'incorrect erc20 implementation',
            'erc20 token transfer to self token address'
        ],
        
        'Implementation': [
            'race condition', 'frontrunning', 'timestamp', 'randomness',
            'cross-function race condition', 'reentrancy race condition',
            'transaction order dependence', 'transaction ordering dependency (tod)',
            'race conditions', 'time manipulation', 'timestamp dependency'
        ],
        
        'External Interaction': [
            'cross-contract', 'bridge', 'injection', 'delegatecall',
            'call to unknown function via fallback()', 'call depth attack',
            'unchecked call return value', 'unchecked return values',
            'mishandled exceptions', 'forcibly sending ether to a contract'
        ],
        
        'Storage/State': [
            'storage', 'uninitialized', 'state manipulation',
            'uninitialized storage pointer', 'shadowing state variables',
            'write to arbitrary storage location', 'unpredictable state'
        ],
        
        'Gas-Related': [
            'gas', 'dos', 'denial of service', 'dos with block gas limit',
            'dos with failed call', 'dos with unbounded operations',
            'dos with unexpected revert', 'ethereum gasless send',
            'insufficient gas griefing', 'message call with hardcoded gas amount',
            'stack size limit', 'underpriced opcodes'
        ],
        
        'Ethereum': [
            'ethereum', 'eth', 'erc20', 'erc-20', 'erc721', 'erc-721', 'smart contract',
            'the ethereum black valentines day vulnerability', 'frozen ether',
            'ether lost in transfer', 'ether lost to orphan address'
        ],
        
        'Cryptographic': [
            'cryptographic attack', 'private key prediction', 'length extension attack',
            'hash collision attack', 'hash collisions with multiple variable length arguments',
            'weak sources of randomness from chain attributes', 'keeping secrets',
            'unencrypted private data on-chain', 'lack of proper signature verification',
            'signature malleability', 'missing protection against signature replay attacks'
        ],
        
        'Transaction': [
            'transaction replay attack', 'transaction malleability attack',
            'time-locked transaction attack', 'short address attack'
        ],
        
        'Consensus': [
            '51% attack', 'finney attack', 'vector76 attack', 'alternative historical attack',
            'grinding attack', 'coin age accumulation attack', 'selfing mining',
            'block double production', 'long range attack', 'bribery attack'
        ],
        
        'Network': [
            'sybil attack', 'eclipse attack', 'eavesdropping attack',
            'denial of service attack', 'bgp hijack attack', 'alien attack',
            'timejacking', 'http input attack', 'cross-domain phishing attack'
        ],
        
        'Governance': [
            'liveness denial', 'censorship'
        ],
        
        'Frontend': [
            'blind signing', 'false top-up attack', 'default fallback address attack'
        ],
        
        'Compiler/Code': [
            'use of outdated compiler version', 'arbitrary jump with function type variable',
            'block values as a proxy for time', 'floating pragma', 'presence of unused variables',
            'right-to-left-override control character (u+202e)', 'source code unavailable for review',
            'typecasts', 'upgradeable contract', 'usage of continue in do-while',
            'use of deprecated solidity functions', 'vulnerabilities in virtual machines',
            'incorrect function state mutability'
        ]
    
    }
    
    url = "https://rekt.news/leaderboard/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        print(f"Fetching data from {url}...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all hack entries
        entries = soup.select('div.leaderboard-row')
        if not entries:
            entries = soup.select('.leaderboard-row')
        
        print(f"Found {len(entries)} hack entries")
        
        data = []
        for entry in entries:
            try:
                # Extract title
                title_elem = entry.select_one('a')
                title = title_elem.get_text(strip=True) if title_elem else "Unknown"
                
                # Extract details
                details_elem = entry.select_one('.leaderboard-row-details')
                if not details_elem:
                    details_elem = entry.select_one('div[class*="details"]')
                
                details_text = details_elem.get_text(strip=True) if details_elem else ""
                
                # Extract cash value
                cash = extract_cash_amount(details_text)
                
                # Extract date
                date = None
                if '|' in details_text:
                    date_str = details_text.split('|')[-1].strip()
                    date = convert_date_format(date_str)
                
                # Get hack type
                hack_type = get_hack_type(title, vulnerability_types)
                
                # Handle special cases for known hacks
                if 'ronin' in title.lower():
                    cash = 625000000
                elif 'poly network' in title.lower():
                    cash = 611000000
                elif 'wormhole' in title.lower():
                    cash = 326000000
                elif 'mango markets' in title.lower():
                    cash = 114000000
                elif 'beanstalk' in title.lower():
                    cash = 182000000
                
                data.append({
                    'title': title,
                    'cash': cash,
                    'date': date,
                    'hack_type': hack_type
                })
                
            except Exception as e:
                print(f"Error processing entry: {e}")
                continue
                
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"Error scraping leaderboard: {e}")
        return pd.DataFrame()

def identify_hacker_groups(hack_data_df):
    """
    Use unsupervised ML to identify potential hacker groups behind hacks
    """
    print("Starting hacker group identification...")
    
    # Check if input DataFrame is valid
    if hack_data_df is None or hack_data_df.empty:
        print("Error: Input DataFrame is None or empty")
        return pd.DataFrame(columns=['title', 'hacker_group', 'confidence'])
    
    # Print input DataFrame info for debugging
    print(f"Input DataFrame has {len(hack_data_df)} rows and these columns: {hack_data_df.columns.tolist()}")
    
    # Define hacker group characteristics
    hacker_groups = {
        'Lazarus Group': {
            'techniques': ['spear-phishing', 'social engineering', 'fake job', 'money laundering', 
                          'crypto mixer', 'cross-chain', 'ransomware'],
            'targets': ['bank', 'cryptocurrency', 'exchange', 'financial'],
            'tools': ['backdoor', 'trojan']
        },
        'APT38': {
            'techniques': ['fraudulent transaction', 'persistence', 'long-term'],
            'targets': ['bank', 'financial', 'swift', 'payment'],
            'tools': ['backdoor']
        },
        'Kimsuky': {
            'techniques': ['spear-phishing', 'information gathering', 'espionage'],
            'targets': ['cryptocurrency', 'financial', 'intelligence'],
            'tools': ['spyware', 'keylogger']
        },
        'AndAriel': {
            'techniques': ['watering hole', 'spear-phishing', 'supply chain'],
            'targets': ['government', 'defense', 'economic'],
            'tools': ['aryan', 'gh0st rat', 'rifdoor']
        }
    }
    
    # Load hack text data from pickle files
    hack_texts = {}
    output_dir = r'C:\Users\losan\Crypto_Hacks\output_crypto'
    
    if os.path.exists(output_dir):
        print(f"Reading files from {output_dir}...")
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(output_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        content = pickle.load(f)
                        # Convert content to string
                        if isinstance(content, str):
                            text_content = content.lower()
                        else:
                            text_content = str(content).lower()
                        
                        hack_name = os.path.splitext(filename)[0]
                        hack_texts[hack_name] = text_content
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    
    if not hack_texts:
        print("No hack text files found. Cannot proceed with analysis.")
        hack_data_df['hacker_group'] = 'Unknown'
        hack_data_df['confidence'] = 0.0
        return hack_data_df
    
    print(f"Successfully read {len(hack_texts)} hack files.")
    
    # Create a DataFrame with hack texts
    text_df = pd.DataFrame(list(hack_texts.items()), columns=['hack_name', 'text'])
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(text_df['text'])
    
    # Dimensionality Reduction
    n_components = min(10, len(text_df))
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())
    
    # K-means Clustering
    n_clusters = min(5, len(text_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)
    
    # Add cluster information to the DataFrame
    text_df['cluster'] = clusters
    
    # Score each hack against each hacker group
    group_scores = {}
    
    def score_hack_against_group(hack_text, group_info):
        score = 0
        
        # Check for techniques
        for technique in group_info['techniques']:
            if technique in hack_text:
                score += 2
        
        # Check for targets
        for target in group_info['targets']:
            if target in hack_text:
                score += 2
        
        # Check for tools
        for tool in group_info['tools']:
            if tool in hack_text:
                score += 3
        
        # Normalize score
        total_items = (
            len(group_info['techniques']) * 2 + 
            len(group_info['targets']) * 2 + 
            len(group_info['tools']) * 3
        )
        
        return score / total_items if total_items > 0 else 0
    
    for hack_name, hack_text in hack_texts.items():
        group_scores[hack_name] = {}
        
        for group_name, group_info in hacker_groups.items():
            score = score_hack_against_group(hack_text, group_info)
            group_scores[hack_name][group_name] = score
            
            # Print high-scoring matches
            if score > 0.1:
                print(f"Match: {hack_name} -> {group_name} with score {score:.3f}")
    
    # Convert scores to DataFrame
    scores_df = pd.DataFrame(group_scores).T
    
    # Find most likely group for each hack
    scores_df['confidence'] = scores_df.max(axis=1)
    scores_df['most_likely_group'] = scores_df.iloc[:, :-1].idxmax(axis=1)
    
    # Apply threshold
    threshold = 0.01  # Very low threshold
    scores_df.loc[scores_df['confidence'] < threshold, 'most_likely_group'] = 'Unknown'
    
    # Reset index and rename columns
    scores_df = scores_df.reset_index().rename(columns={'index': 'hack_name'})
    
    # Create mapping dictionaries
    hack_to_group = dict(zip(scores_df['hack_name'], scores_df['most_likely_group']))
    hack_to_confidence = dict(zip(scores_df['hack_name'], scores_df['confidence']))
    
    # Function to match input titles to hack names
    def find_best_match(title):
        if title is None or not isinstance(title, str):
            return 'Unknown', 0.0
        
        # Normalize the title
        title_lower = str(title).lower()
        title_base = re.sub(r'[^a-z0-9]', '', title_lower)
        
        # Direct keyword matching
        if any(kw in title_lower for kw in ['ronin', 'wormhole', 'harmony', 'bridge']):
            return 'Lazarus Group', 0.8
        elif any(kw in title_lower for kw in ['mango', 'platypus', 'beanstalk']):
            return 'APT38', 0.8
        
        # Find best matching hack name
        best_match = None
        best_score = 0
        
        for hack_name in hack_to_group:
            # Extract base name (before any ___ or numbers)
            hack_parts = hack_name.split('___')
            hack_base = hack_parts[0].lower() if hack_parts else hack_name.lower()
            hack_base = re.sub(r'[^a-z0-9]', '', hack_base)
            
            # Simple substring matching
            if hack_base in title_base or title_base in hack_base:
                score = 0.8
            else:
                # Word overlap matching
                title_words = set(re.findall(r'\w+', title_lower))
                hack_words = set(re.findall(r'\w+', hack_base))
                common_words = title_words.intersection(hack_words)
                
                if common_words:
                    score = len(common_words) / min(len(title_words), len(hack_words))
                else:
                    score = 0
            
            if score > best_score:
                best_score = score
                best_match = hack_name
        
        # Use very lenient threshold
        if best_score > 0.01 and best_match:
            group = hack_to_group.get(best_match, 'Unknown')
            confidence = hack_to_confidence.get(best_match, 0.0)
            print(f"Matched '{title}' to '{best_match}': {group} (score: {best_score:.2f})")
            return group, confidence
        
        return 'Unknown', 0.0
    
    # Apply matching to input DataFrame
    print(f"Processing {len(hack_data_df)} input rows...")
    
    # Create new columns if they don't exist
    if 'hacker_group' not in hack_data_df.columns:
        hack_data_df['hacker_group'] = 'Unknown'
    if 'confidence' not in hack_data_df.columns:
        hack_data_df['confidence'] = 0.0
    
    # Match each title
    for idx, row in hack_data_df.iterrows():
        title = row.get('title', '')
        group, confidence = find_best_match(title)
        hack_data_df.at[idx, 'hacker_group'] = group
        hack_data_df.at[idx, 'confidence'] = confidence
    
    print(f"Completed processing. Found {(hack_data_df['hacker_group'] != 'Unknown').sum()} matches.")
    return hack_data_df

def main():
    print("Starting main function...")
    # Set environment variable to avoid KMeans memory leak warning
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Get leaderboard data
    print("Scraping leaderboard data...")
    leaderboard_data = scrape_rekt_leaderboard()
    
    if leaderboard_data.empty:
        print("No data retrieved from leaderboard. Check your internet connection.")
        return
    
    print(f"Leaderboard data columns: {leaderboard_data.columns.tolist()}")
    print(f"Leaderboard data shape: {leaderboard_data.shape}")
    
    # Normalize cash values
    print("Normalizing cash values...")
    if 'cash' in leaderboard_data.columns and not leaderboard_data['cash'].empty and leaderboard_data['cash'].sum() > 0:
        leaderboard_data['normalized_cash'] = minmax_scale(leaderboard_data[['cash']])
    else:
        leaderboard_data['normalized_cash'] = 0.0
        print("Warning: 'cash' column not found or contains only zeros")
    
    # Clean titles for better matching
    leaderboard_data['clean_title'] = leaderboard_data['title'].apply(
        lambda x: x.lower().replace('rekt', '').replace('-', '').strip()
    )
    
    # Try to load brutality analysis data
    print("Attempting to load brutality analysis data...")
    # Define lb_features_dict if it doesn't exist
    lb_features_dict = {}
    
    # Try to load from a file if it exists
    features_file = 'lb_features.pkl'
    if os.path.exists(features_file):
        print(f"Loading features from {features_file}...")
        with open(features_file, 'rb') as f:
            lb_features_dict = pickle.load(f)
        print(f"Loaded {len(lb_features_dict)} items from features file")
    else:
        print(f"Features file {features_file} not found")
    
    if lb_features_dict:
        print("Creating DataFrame from brutality analysis data...")
        text_data = pd.DataFrame({
            'title': list(lb_features_dict.keys()),
            'total_words': [x[0] for x in lb_features_dict.values()],
            'brutal_words': [x[1] for x in lb_features_dict.values()],
            'brutal_prop': [x[2] for x in lb_features_dict.values()]
        })
        
        print(f"Text data columns: {text_data.columns.tolist()}")
        print(f"Text data shape: {text_data.shape}")
        
        # Clean text data titles for better matching
        text_data['clean_title'] = text_data['title'].apply(
            lambda x: x.lower().replace('rekt', '').replace('-', '').strip()
        )
        
        # Merge on clean titles
        print("Merging data on clean titles...")
        final_df = pd.merge(
            text_data.drop('clean_title', axis=1),
            leaderboard_data.drop('clean_title', axis=1),
            on='title',
            how='left'
        )
        
        print(f"Merged data columns: {final_df.columns.tolist()}")
        print(f"Merged data shape: {final_df.shape}")
        
        # If the merge didn't work well, try fuzzy matching
        if 'cash' not in final_df.columns or final_df['cash'].isna().all() or final_df['cash'].sum() == 0:
            print("Direct merge failed. Trying fuzzy matching...")
            
            # Create a mapping of clean titles from leaderboard to full data
            leaderboard_map = {row['clean_title']: row for _, row in leaderboard_data.iterrows()}
            
            # Function to find best match
            def find_best_match(clean_title):
                best_match = None
                best_score = 0
                
                for lb_title in leaderboard_map.keys():
                    # Simple matching score - count of common words
                    score = sum(word in lb_title for word in clean_title.split())
                    if score > best_score:
                        best_score = score
                        best_match = lb_title
                
                return leaderboard_map.get(best_match, {})
            
            # Apply fuzzy matching
            matched_data = []
            for _, row in text_data.iterrows():
                match_data = find_best_match(row['clean_title'])
                matched_data.append({
                    'title': row['title'],
                    'total_words': row['total_words'],
                    'brutal_words': row['brutal_words'],
                    'brutal_prop': row['brutal_prop'],
                    'cash': match_data.get('cash', 0),
                    'date': match_data.get('date', ''),
                    'hack_type': match_data.get('hack_type', 'Unknown'),
                    'normalized_cash': match_data.get('normalized_cash', 0)
                })
            
            final_df = pd.DataFrame(matched_data)
            print(f"Fuzzy matched data columns: {final_df.columns.tolist()}")
            print(f"Fuzzy matched data shape: {final_df.shape}")
    else:
        print("No brutality analysis data found. Using only leaderboard data.")
        final_df = leaderboard_data.copy()
        # Add empty columns for brutality data
        final_df['total_words'] = 0
        final_df['brutal_words'] = 0
        final_df['brutal_prop'] = 0.0
    
    # Identify potential hacker groups - IMPROVED VERSION
    print("Identifying potential hacker groups...")
    
    # First identify groups using our ML approach
    final_df = identify_hacker_groups(final_df)
    
    # IMPROVEMENT: Add additional keyword-based matching for more matches
    print("Adding additional keyword-based matching...")
    for idx, row in final_df.iterrows():
        if row['hacker_group'] == 'Unknown':
            title_lower = str(row.get('title', '')).lower()
            
            # Additional keyword matching for APT38
            if any(kw in title_lower for kw in ['defi', 'protocol', 'swap', 'finance', 'dao', 'token', 
                                               'yield', 'farm', 'staking', 'liquidity']):
                final_df.at[idx, 'hacker_group'] = 'APT38'
                final_df.at[idx, 'confidence'] = 0.4
            
            # Additional keyword matching for Lazarus Group
            elif any(kw in title_lower for kw in ['exchange', 'cross-chain', 'bridge', 'wallet', 
                                                 'custody', 'asset', 'transfer']):
                final_df.at[idx, 'hacker_group'] = 'Lazarus Group'
                final_df.at[idx, 'confidence'] = 0.4
            
            # Additional keyword matching for Kimsuky
            elif any(kw in title_lower for kw in ['phishing', 'social', 'email', 'credential']):
                final_df.at[idx, 'hacker_group'] = 'Kimsuky'
                final_df.at[idx, 'confidence'] = 0.4
    
    print(f"After additional matching: {(final_df['hacker_group'] != 'Unknown').sum()} matches")
    print(f"After hacker group identification, columns: {final_df.columns.tolist()}")
    print(f"After hacker group identification, shape: {final_df.shape}")
    
    # Make sure all required columns exist
    required_columns = ['title', 'cash', 'date', 'total_words', 'brutal_words', 
                        'brutal_prop', 'hack_type', 'hacker_group', 'confidence', 'normalized_cash']
    
    print("Checking for required columns...")
    for col in required_columns:
        if col not in final_df.columns:
            print(f"Adding missing column: {col}")
            if col == 'cash':
                final_df[col] = 0
            elif col == 'date':
                final_df[col] = ''
            elif col == 'hack_type':
                final_df[col] = 'Unknown'
            elif col == 'hacker_group':
                final_df[col] = 'Unknown'
            elif col == 'confidence':
                final_df[col] = 0.0
            elif col == 'normalized_cash':
                final_df[col] = 0.0
            else:
                final_df[col] = 0
    
    # Fill NaN values
    print("Filling NaN values...")
    final_df['cash'] = final_df['cash'].fillna(0)
    final_df['normalized_cash'] = final_df['normalized_cash'].fillna(0)
    final_df['hack_type'] = final_df['hack_type'].fillna('Unknown')
    final_df['date'] = final_df['date'].fillna('')
    final_df['total_words'] = final_df['total_words'].fillna(0)
    final_df['brutal_words'] = final_df['brutal_words'].fillna(0)
    final_df['brutal_prop'] = final_df['brutal_prop'].fillna(0.0)
    
    # Add cash_formatted column
    print("Adding cash_formatted column...")
    final_df['cash_formatted'] = final_df['cash'].apply(format_cash_value)
    
    # Reorder columns to match the expected format
    columns = ['title', 'cash_formatted', 'date', 'total_words', 'brutal_words', 
            'brutal_prop', 'hack_type', 'hacker_group', 'confidence', 'normalized_cash']
    
    # Ensure all columns exist before reordering
    print("Reordering columns...")
    available_columns = []
    for col in columns:
        if col in final_df.columns:
            available_columns.append(col)
        else:
            print(f"Warning: Column {col} is missing")
    
    # Only include columns that exist
    final_df = final_df[available_columns]
    
    # Save results
    #output_file = 'hack_analysis_enhanced.csv'
    #print(f"Saving results to {output_file}...")
    #final_df.to_csv(output_file, index=False)
    #print(f"Results saved to {output_file}")
    
    # Save formatted version
    formatted_file = 'hack_analysis_formatted.csv'
    print(f"Saving formatted version to {formatted_file}...")
    formatted_df = final_df.copy()
    formatted_df.to_csv(formatted_file, index=False)
    print(f"Formatted data saved to {formatted_file}")


    
    # Display top 10 hacks
    print("\nTop 10 Hacks by Brutality and Stolen Amount:")
    sort_columns = [col for col in ['brutal_prop', 'cash'] if col in final_df.columns]
    
    if sort_columns:
        top_10 = final_df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(10)
        print(top_10)  # Show all columns
    else:
        print("No sort columns available, showing first 10 rows:")
        print(final_df.head(10))  # Show all columns
    
    # Also print the columns to verify
    print("\nAll columns in the DataFrame:")
    print(final_df.columns.tolist())
    
    print("Main function completed successfully")

if __name__ == "__main__":
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)
    
    main()


# ## Social Network Analysis

# In[32]:


import os
import re
import pickle
import networkx as nx
import requests
import matplotlib.pyplot as plt
import pandas as pd
import time
import json

# IMPORTANT: You need to get a free Etherscan API key
# Sign up at https://etherscan.io/apis and add your key here
ETHERSCAN_API_KEY = "7R4QA3FCPSE1YPJBV2KT327ZDQYX15KII8"

def find_ethereum_addresses_by_hack(directory):
    """
    Search through files in the specified directory to find Ethereum addresses
    organized by hack name
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing files to search
        
    Returns:
    --------
    hack_addresses : dict
        Dictionary mapping hack names to lists of Ethereum addresses
    """
    print(f"Searching for Ethereum addresses by hack in {directory}...")
    
    # Regular expression for Ethereum addresses (0x followed by 40 hex characters)
    eth_address_pattern = re.compile(r'0x[a-fA-F0-9]{40}')
    
    hack_addresses = {}
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return hack_addresses
    
    # Count of files processed and addresses found
    files_processed = 0
    total_addresses = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        try:
            # Get hack name from filename (remove extension)
            hack_name = os.path.splitext(filename)[0]
            
            # Handle different file types
            if filename.endswith('.pkl'):
                # Try to read pickle files
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    
                    # Convert content to string if it's not already
                    if not isinstance(content, str):
                        content = str(content)
                    
            elif filename.endswith(('.txt', '.csv', '.json')):
                # Read text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                # Skip other file types
                continue
                
            # Find all Ethereum addresses in the content
            found_addresses = eth_address_pattern.findall(content)
            
            # Add to our dictionary, mapping hack name to addresses
            if found_addresses:
                if hack_name not in hack_addresses:
                    hack_addresses[hack_name] = []
                hack_addresses[hack_name].extend(found_addresses)
                total_addresses += len(found_addresses)
            
            files_processed += 1
            if files_processed % 10 == 0:
                print(f"Processed {files_processed} files, found {total_addresses} addresses across {len(hack_addresses)} hacks")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Remove duplicates from each hack's address list
    for hack_name in hack_addresses:
        hack_addresses[hack_name] = list(set(hack_addresses[hack_name]))
    
    print(f"Completed search. Processed {files_processed} files and found {total_addresses} addresses across {len(hack_addresses)} hacks")
    return hack_addresses

def get_transactions_for_address(address, max_transactions=100):
    """
    Get transactions for an Ethereum address using Etherscan API
    
    Parameters:
    -----------
    address : str
        Ethereum address to get transactions for
    max_transactions : int
        Maximum number of transactions to retrieve
        
    Returns:
    --------
    transactions : list
        List of transaction dictionaries
    """
    print(f"Fetching transactions for address: {address}")
    transactions = []
    
    if not ETHERSCAN_API_KEY:
        print("ERROR: Etherscan API key is required. Please get a free API key from https://etherscan.io/apis")
        return transactions
    
    try:
        # Normal transactions
        api_url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset={max_transactions}&sort=desc&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(api_url, timeout=10)
        data = response.json()
        
        if data.get('status') == '1' and 'result' in data:
            for tx in data['result']:
                transactions.append({
                    'hash': tx.get('hash', ''),
                    'from': tx.get('from', ''),
                    'to': tx.get('to', ''),
                    'value': float(tx.get('value', '0')) / 1e18,  # Convert from wei to ETH
                    'timestamp': tx.get('timeStamp', ''),
                    'type': 'normal'
                })
        else:
            print(f"API Error: {data.get('message', 'Unknown error')}")
        
        # Token transactions (ERC-20)
        api_url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&page=1&offset={max_transactions}&sort=desc&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(api_url, timeout=10)
        data = response.json()
        
        if data.get('status') == '1' and 'result' in data:
            for tx in data['result']:
                transactions.append({
                    'hash': tx.get('hash', ''),
                    'from': tx.get('from', ''),
                    'to': tx.get('to', ''),
                    'value': float(tx.get('value', '0')) / (10 ** int(tx.get('tokenDecimal', '18'))),
                    'timestamp': tx.get('timeStamp', ''),
                    'token': tx.get('tokenSymbol', ''),
                    'type': 'token'
                })
        else:
            print(f"API Error for token transactions: {data.get('message', 'Unknown error')}")
        
        print(f"Found {len(transactions)} transactions via API")
        
    except Exception as e:
        print(f"Error using Etherscan API: {e}")
    
    return transactions

def create_transaction_network(hack_name, addresses, max_addresses=5, max_transactions=20):
    """
    Create a transaction network for a specific hack using real Etherscan data
    
    Parameters:
    -----------
    hack_name : str
        Name of the hack
    addresses : list
        List of Ethereum addresses associated with the hack
    max_addresses : int
        Maximum number of addresses to include in the network
    max_transactions : int
        Maximum number of transactions to include per address
        
    Returns:
    --------
    G : networkx.Graph
        Network graph of transactions
    all_transactions : list
        List of all transactions
    """
    print(f"Creating transaction network for {hack_name} with Etherscan data...")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Limit the number of addresses to analyze
    if len(addresses) > max_addresses:
        print(f"Limiting analysis to {max_addresses} addresses out of {len(addresses)}")
        addresses = addresses[:max_addresses]
    
    # Add all addresses as nodes first
    for addr in addresses:
        G.add_node(addr, type='address', label=addr[:10])
    
    # Get transactions for each address
    all_transactions = []
    processed_hashes = set()  # To avoid duplicate transactions
    
    for addr in addresses:
        # Respect rate limits
        time.sleep(1)
        
        # Get transactions for this address
        addr_transactions = get_transactions_for_address(addr, max_transactions=max_transactions)
        
        # Process each transaction
        for tx in addr_transactions:
            # Skip if we've already processed this transaction
            if tx['hash'] in processed_hashes:
                continue
                
            processed_hashes.add(tx['hash'])
            all_transactions.append(tx)
            
            # Add nodes if they don't exist
            from_addr = tx['from']
            to_addr = tx['to']
            
            if from_addr not in G:
                G.add_node(from_addr, type='address' if from_addr in addresses else 'external', 
                           label=from_addr[:10])
            
            if to_addr not in G:
                G.add_node(to_addr, type='address' if to_addr in addresses else 'external',
                           label=to_addr[:10])
            
            # Add edge with transaction details
            G.add_edge(from_addr, to_addr, 
                       weight=tx['value'],
                       tx_hash=tx['hash'],
                       type=tx.get('type', 'normal'),
                       token=tx.get('token', 'ETH'))
    
    # Add labels for known addresses
    known_addresses = {
        "0xdAC17F958D2ee523a2206206994597C13D831ec7": "USDT",
        "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0": "MATIC",
        "0x6B175474E89094C44Da98b954EedeAC495271d0F": "DAI",
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": "USDC",
        "0x3883f5e181fccaF8410FA61e12b59BAd963fb645": "Tornado Cash",
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "WETH"
    }
    
    for addr, label in known_addresses.items():
        if addr in G.nodes():
            G.nodes[addr]['label'] = label
    
    print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, all_transactions

def calculate_network_metrics(G):
    """
    Calculate network metrics for a graph
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph
        
    Returns:
    --------
    metrics : dict
        Dictionary of network metrics
    """
    metrics = {}
    
    # Number of nodes
    metrics['N_d'] = G.number_of_nodes()
    
    # Number of edges
    metrics['E_d'] = G.number_of_edges()
    
    # Edge betweenness centrality
    if metrics['E_d'] > 0:
        try:
            edge_betweenness = nx.edge_betweenness_centrality(G)
            metrics['E_d_E'] = sum(edge_betweenness.values()) / len(edge_betweenness)
        except:
            metrics['E_d_E'] = 0
    else:
        metrics['E_d_E'] = 0
    
    # Node centrality (degree centrality)
    degree_centrality = nx.degree_centrality(G)
    metrics['C_N'] = sum(degree_centrality.values()) / len(degree_centrality)
    
    # Betweenness centrality
    if metrics['N_d'] > 1:
        try:
            betweenness_centrality = nx.betweenness_centrality(G)
            metrics['B_tw_N'] = sum(betweenness_centrality.values()) / len(betweenness_centrality)
        except:
            metrics['B_tw_N'] = 0
    else:
        metrics['B_tw_N'] = 0
    
    # Closeness centrality
    try:
        if nx.is_strongly_connected(G):
            closeness_centrality = nx.closeness_centrality(G)
            metrics['C'] = sum(closeness_centrality.values()) / len(closeness_centrality)
        else:
            # For disconnected graphs, calculate on largest component
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if subgraph.number_of_nodes() > 1:
                closeness_centrality = nx.closeness_centrality(subgraph)
                metrics['C'] = sum(closeness_centrality.values()) / len(closeness_centrality)
            else:
                metrics['C'] = 0
    except:
        metrics['C'] = 0
    
    # Eigenvector centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        metrics['EC_C'] = sum(eigenvector_centrality.values()) / len(eigenvector_centrality)
    except:
        metrics['EC_C'] = 0
    
    return metrics

def visualize_transaction_network(G, hack_name):
    """
    Visualize a transaction network similar to the provided image
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph
    hack_name : str
        Name of the hack for the title
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=(12, 10))
    
    # Use spring layout with adjusted parameters for better spacing
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Node sizes based on degree centrality
    degree_cent = nx.degree_centrality(G)
    node_sizes = [300 + 2000 * degree_cent[node] for node in G.nodes()]
    
    # Node colors based on type
    node_colors = ['#1f77b4' if G.nodes[node].get('type') == 'address' else '#ff7f0e' for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Edge widths based on value
    edge_values = [G[u][v].get('weight', 0.1) for u, v in G.edges()]
    max_value = max(edge_values) if edge_values else 1
    edge_widths = [0.5 + 3 * (val / max_value) for val in edge_values]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, arrows=True)
    
    # Draw labels with the node's label attribute
    labels = {node: G.nodes[node].get('label', '') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    plt.title(hack_name, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    output_file = f"{hack_name.replace(' ', '_')}_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {output_file}")
    
    plt.show()

def main():
    """
    Main function to find Ethereum addresses by hack and create transaction networks
    """
    # Check for API key
    if not ETHERSCAN_API_KEY:
        print("ERROR: Etherscan API key is required.")
        print("Please get a free API key from https://etherscan.io/apis")
        print("Add your API key to the ETHERSCAN_API_KEY variable at the top of the script.")
        return
    
    # Directory to search for Ethereum addresses
    directory = r'C:\Users\losan\Crypto_Hacks\output_crypto'
    
    # Find Ethereum addresses organized by hack
    hack_addresses = find_ethereum_addresses_by_hack(directory)
    
    if not hack_addresses:
        print("No Ethereum addresses found for any hacks")
        return
    
    # Display found hacks and address counts
    print("\nFound Ethereum addresses for the following hacks:")
    for i, (hack_name, addresses) in enumerate(hack_addresses.items(), 1):
        print(f"{i}. {hack_name}: {len(addresses)} addresses")
    
    # Create a results DataFrame for metrics
    results = pd.DataFrame(columns=['hack_name', 'N_d', 'E_d', 'E_d_E', 'C_N', 'B_tw_N', 'C', 'EC_C'])
    
    # Process each hack (limit to top 3 by address count to avoid rate limiting)
    top_hacks = sorted(hack_addresses.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    
    for hack_name, addresses in top_hacks:
        print(f"\nProcessing {hack_name} with {len(addresses)} addresses")
        
        try:
            # Create transaction network with real Etherscan data
            G, transactions = create_transaction_network(hack_name, addresses)
            
            # Skip if no transactions were found
            if not transactions:
                print(f"No transactions found for {hack_name}. Skipping.")
                continue
            
            # Calculate network metrics
            metrics = calculate_network_metrics(G)
            
            # Add to results DataFrame
            results = pd.concat([results, pd.DataFrame({
                'hack_name': [hack_name],
                'N_d': [metrics.get('N_d', 0)],
                'E_d': [metrics.get('E_d', 0)],
                'E_d_E': [metrics.get('E_d_E', 0)],
                'C_N': [metrics.get('C_N', 0)],
                'B_tw_N': [metrics.get('B_tw_N', 0)],
                'C': [metrics.get('C', 0)],
                'EC_C': [metrics.get('EC_C', 0)]
            })], ignore_index=True)
            
            # Visualize the network
            visualize_transaction_network(G, hack_name)
            
            # Save transactions to CSV
            tx_df = pd.DataFrame(transactions)
            if not tx_df.empty:
                tx_file = f"{hack_name.replace(' ', '_')}_transactions.csv"
                tx_df.to_csv(tx_file, index=False)
                print(f"Transaction data saved to {tx_file}")
            
            # Be nice to Etherscan
            time.sleep(3)
            
        except Exception as e:
            print(f"Error processing {hack_name}: {e}")
    
    # Save metrics results
    if not results.empty:
        results_file = "hack_network_metrics.csv"
        results.to_csv(results_file, index=False)
        print(f"\nSaved network metrics for {len(results)} hacks to {results_file}")
        
        # Display metrics summary
        print("\nNetwork Metrics Summary:")
        print(results)
        
        # Create a comparison visualization of metrics
        plt.figure(figsize=(14, 10))
        
        # Plot metrics for each hack
        metrics_to_plot = ['N_d', 'E_d', 'C_N', 'B_tw_N', 'C', 'EC_C']
        
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(3, 2, i)
            plt.bar(results['hack_name'], results[metric])
            plt.title(f"{metric} by Hack")
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig("hack_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()


# ## Produce JSONS

# In[33]:


# CSV and PNG to JSON Conversion
import csv
import json
import os
import base64
import glob
from pathlib import Path

def convert_csv_to_json(csv_file_path, json_file_path, numeric_fields=None):
    """
    Convert a CSV file to JSON format.
    
    Args:
        csv_file_path (str): Path to the CSV file
        json_file_path (str): Path to save the JSON file
        numeric_fields (dict): Dictionary mapping field names to their types ('int' or 'float')
    
    Returns:
        list: The data as a list of dictionaries
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None
    
    # Read the CSV file
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Convert numeric fields to appropriate types if specified
            if numeric_fields:
                for field, field_type in numeric_fields.items():
                    try:
                        if field in row and row[field]:
                            if field_type == 'int':
                                row[field] = int(row[field])
                            elif field_type == 'float':
                                row[field] = float(row[field])
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Error converting field '{field}' in row: {e}")
            data.append(row)
    
    # Write the data to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Successfully converted '{csv_file_path}' to '{json_file_path}'.")
    print(f"Converted {len(data)} records.")
    
    return data

def convert_png_to_base64(png_file_path):
    """
    Convert a PNG file to a base64-encoded string.
    
    Args:
        png_file_path (str): Path to the PNG file
    
    Returns:
        str: Base64-encoded string of the PNG file
    """
    try:
        with open(png_file_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error encoding {png_file_path}: {e}")
        return None

def convert_png_files_to_json(png_directory, json_file_path, pattern="*.png"):
    """
    Convert all PNG files in a directory to a JSON file with base64-encoded strings.
    
    Args:
        png_directory (str): Directory containing PNG files
        json_file_path (str): Path to save the JSON file
        pattern (str): Pattern to match PNG files
    
    Returns:
        dict: Dictionary mapping filenames to base64-encoded strings
    """
    # Check if the directory exists
    if not os.path.exists(png_directory):
        print(f"Error: Directory '{png_directory}' not found.")
        return None
    
    # Find all PNG files in the directory
    png_files = glob.glob(os.path.join(png_directory, pattern))
    
    if not png_files:
        print(f"No PNG files found in '{png_directory}' matching pattern '{pattern}'.")
        return None
    
    # Convert each PNG file to base64
    png_data = {}
    for png_file in png_files:
        filename = os.path.basename(png_file)
        encoded_string = convert_png_to_base64(png_file)
        if encoded_string:
            png_data[filename] = encoded_string
    
    # Write the data to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(png_data, json_file, indent=4)
    
    print(f"Successfully converted {len(png_data)} PNG files to '{json_file_path}'.")
    
    return png_data

# 1. Convert hack_analysis_formatted.csv
print("Converting hack analysis data...")
hack_csv_path = 'hack_analysis_formatted.csv'
hack_json_path = 'hack_analysis.json'

# Define numeric fields for hack analysis data
hack_numeric_fields = {
    'total_words': 'int',
    'brutal_words': 'int',
    'brutal_prop': 'float',
    'confidence': 'float',
    'normalized_cash': 'float'
}

# Convert hack analysis CSV to JSON
hack_data = convert_csv_to_json(hack_csv_path, hack_json_path, hack_numeric_fields)

# Display first few records to verify
if hack_data:
    print("\nFirst 3 records from the hack analysis data:")
    for i, record in enumerate(hack_data[:3]):
        print(f"\nRecord {i+1}:")
        for key, value in record.items():
            print(f"  {key}: {value}")

# 2. Convert output_crypto_transactions.csv
print("\nConverting crypto transactions data...")
transactions_csv_path = 'output_crypto_transactions.csv'
transactions_json_path = 'crypto_transactions.json'

# Define numeric fields for transaction data
transaction_numeric_fields = {
    'value': 'float',
    'timestamp': 'int'
}

# Convert transactions CSV to JSON
transaction_data = convert_csv_to_json(transactions_csv_path, transactions_json_path, transaction_numeric_fields)

# Display first few records to verify
if transaction_data:
    print("\nFirst 3 records from the transaction data:")
    for i, record in enumerate(transaction_data[:3]):
        print(f"\nRecord {i+1}:")
        for key, value in record.items():
            print(f"  {key}: {value}")

# 3. Convert PNG files to JSON
print("\nConverting PNG files to JSON...")
# You can change this to the directory containing your PNG files
png_directory = '.'  # Current directory, change as needed
png_json_path = 'images.json'

# Convert PNG files to JSON
png_data = convert_png_files_to_json(png_directory, png_json_path)

# Display information about the converted PNG files
if png_data:
    print(f"\nConverted {len(png_data)} PNG files to base64 in JSON:")
    for i, (filename, _) in enumerate(list(png_data.items())[:3]):
        # Only show the filename, not the entire base64 string (too long)
        print(f"  {i+1}. {filename} - {len(png_data[filename])//1024} KB encoded")


# ## Export Jsons to local IPFS  first run command in Ubuntu    ipfs daemon 

# In[34]:


'''
#Export JSON files to IPFS using direct HTTP API calls
import requests
import os
import json
import pandas as pd
from datetime import datetime

def export_to_ipfs_http(file_paths, ipfs_api_url="http://127.0.0.1:5001/api/v0", 
                        create_index=True, index_file="ipfs_index.json"):
    """
    Export files to IPFS using direct HTTP API calls.
    
    Args:
        file_paths (list): List of file paths to export to IPFS
        ipfs_api_url (str): URL of the IPFS API
        create_index (bool): Whether to create an index file with all CIDs
        index_file (str): Path to save the index file
    
    Returns:
        dict: Dictionary mapping file names to their IPFS CIDs
    """
    # Check if files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None
    
    # Check if IPFS daemon is running
    try:
        response = requests.post(f"{ipfs_api_url}/version")
        if response.status_code != 200:
            print(f"Error connecting to IPFS daemon: {response.text}")
            print("Make sure your IPFS daemon is running with 'ipfs daemon' command")
            return None
        
        ipfs_version = response.json().get('Version', 'unknown')
        print(f"Connected to IPFS daemon (version {ipfs_version})")
    except Exception as e:
        print(f"Error connecting to IPFS daemon: {e}")
        print("Make sure your IPFS daemon is running with 'ipfs daemon' command")
        return None
    
    # Upload each file to IPFS
    results = {}
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            print(f"Uploading {file_name} to IPFS...")
            
            # Add the file to IPFS using the HTTP API
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.post(f"{ipfs_api_url}/add", files=files)
                
                if response.status_code != 200:
                    print(f"Error uploading {file_name}: {response.text}")
                    continue
                
                # Parse the response
                res = response.json()
                
                # Get the CID (Content Identifier)
                cid = res.get('Hash')
                
                if not cid:
                    print(f"Error: No CID returned for {file_name}")
                    continue
                
                # Store the result
                results[file_name] = {
                    'cid': cid,
                    'size': res.get('Size', 0),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'gateway_url': f"https://ipfs.io/ipfs/{cid}"
                }
                
                print(f"✅ Successfully uploaded {file_name}")
                print(f"   CID: {cid}")
                print(f"   Gateway URL: https://ipfs.io/ipfs/{cid}")
                print(f"   Local URL: http://localhost:8080/ipfs/{cid}")
                print("--------------------------------------------------")
            
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
    
    # Create an index file with all CIDs
    if create_index and results:
        try:
            with open(index_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Also upload the index file to IPFS
            with open(index_file, 'rb') as file:
                files = {'file': file}
                response = requests.post(f"{ipfs_api_url}/add", files=files)
                
                if response.status_code != 200:
                    print(f"Error uploading index file: {response.text}")
                else:
                    res = response.json()
                    index_cid = res.get('Hash')
                    
                    print(f"Created index file: {index_file}")
                    print(f"Index CID: {index_cid}")
                    print(f"Index Gateway URL: https://ipfs.io/ipfs/{index_cid}")
                    
                    # Add the index to the results
                    results['_index'] = {
                        'cid': index_cid,
                        'size': res.get('Size', 0),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'gateway_url': f"https://ipfs.io/ipfs/{index_cid}"
                    }
            
        except Exception as e:
            print(f"Error creating index file: {e}")
    
    return results

def pin_to_ipfs_http(cids, ipfs_api_url="http://127.0.0.1:5001/api/v0"):
    """
    Pin files to IPFS to ensure they remain available.
    
    Args:
        cids (list): List of CIDs to pin
        ipfs_api_url (str): URL of the IPFS API
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if IPFS daemon is running
        response = requests.post(f"{ipfs_api_url}/version")
        if response.status_code != 200:
            print(f"Error connecting to IPFS daemon: {response.text}")
            return False
        
        success = True
        for cid in cids:
            print(f"Pinning {cid}...")
            response = requests.post(f"{ipfs_api_url}/pin/add?arg={cid}")
            
            if response.status_code != 200:
                print(f"Error pinning {cid}: {response.text}")
                success = False
            else:
                print(f"✅ Successfully pinned {cid}")
        
        return success
    except Exception as e:
        print(f"Error pinning to IPFS: {e}")
        return False

# Files to export to IPFS
files_to_export = [
    'hack_analysis.json',
    'crypto_transactions.json',
    'images.json'  # If you created this from the PNG conversion
]

# Export files to IPFS
print("Exporting files to IPFS...")
ipfs_results = export_to_ipfs_http(files_to_export)

if ipfs_results:
    # Create a DataFrame for better visualization
    ipfs_df = pd.DataFrame([
        {
            'File': file_name,
            'CID': details['cid'],
            'Size (bytes)': details['size'],
            'Timestamp': details['timestamp'],
            'Gateway URL': details['gateway_url']
        }
        for file_name, details in ipfs_results.items()
        if file_name != '_index'  # Exclude the index entry
    ])
    
    # Display the results
    print("\nIPFS Export Summary:")
    display(ipfs_df)
    
    # Pin the files to ensure they remain available
    print("\nPinning files to IPFS...")
    pin_to_ipfs_http([details['cid'] for details in ipfs_results.values()])

'''


# In[36]:





# In[37]:





# ## Export Jsons to Production IPFS Server

# In[35]:


'''
import requests
import os
import json
import pandas as pd
from datetime import datetime

# Pinata API Configuration
PINATA_API_URL = "https://api.pinata.cloud"
PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiIzNzA4YjFlZS0wMDg3LTRjNjUtOWFhOC1jMjUyYWEyMzYyNjYiLCJlbWFpbCI6Iml5ZXIuYXNoaXNoOUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGluX3BvbGljeSI6eyJyZWdpb25zIjpbeyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJGUkExIn0seyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJOWUMxIn1dLCJ2ZXJzaW9uIjoxfSwibWZhX2VuYWJsZWQiOmZhbHNlLCJzdGF0dXMiOiJBQ1RJVkUifSwiYXV0aGVudGljYXRpb25UeXBlIjoic2NvcGVkS2V5Iiwic2NvcGVkS2V5S2V5IjoiZWNjZTBkNDg4ZDNkZjU2YzVjNjYiLCJzY29wZWRLZXlTZWNyZXQiOiJiMTFhMWZmOTFjOGRmMWU5MDM2MzhmNWVlYTdkNTQ1OGU3ZGE0YjJkMjUxYmQ4MTUzMzNkNzRjMDJjZTFiYTU5IiwiZXhwIjoxNzc5NjIyMDE4fQ.8ACFw67zjRigYXdhQ_7zH4FyJU8EBYi27DZRW68iwtI"
HEADERS = {
    "Authorization": f"Bearer {PINATA_JWT}"
}

def upload_to_pinata(file_path):
    """
    Upload a file to Pinata IPFS.
    
    Args:
        file_path (str): Path to the file to upload
    
    Returns:
        dict: Upload result with CID and other metadata
    """
    try:
        file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as file:
            files = {'file': (file_name, file)}
            response = requests.post(
                f"{PINATA_API_URL}/pinning/pinFileToIPFS",
                headers=HEADERS,
                files=files
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json(),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'data': None,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': str(e)
        }

def pin_by_cid(cid, name=None):
    """
    Pin a file by its CID to ensure it remains available.
    
    Args:
        cid (str): The CID of the file to pin
        name (str): Optional name for the pinned file
    
    Returns:
        dict: Result of the pin operation
    """
    try:
        data = {
            'hashToPin': cid,
            'pinataMetadata': {
                'name': name or f'pinned-{cid[:8]}',
                'keyvalues': {
                    'pinned_at': datetime.now().isoformat()
                }
            }
        }
        
        response = requests.post(
            f"{PINATA_API_URL}/pinning/pinByHash",
            headers=HEADERS,
            json=data
        )
        
        if response.status_code == 200:
            return {
                'success': True,
                'data': response.json(),
                'error': None
            }
        else:
            return {
                'success': False,
                'data': None,
                'error': f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': str(e)
        }

def export_to_ipfs(file_paths, create_index=True, index_file="ipfs_index.json"):
    """
    Export files to IPFS using Pinata API.
    
    Args:
        file_paths (list): List of file paths to export to IPFS
        create_index (bool): Whether to create an index file with all CIDs
        index_file (str): Path to save the index file
    
    Returns:
        dict: Dictionary mapping file names to their IPFS CIDs
    """
    # Check if files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None
    
    # Upload each file to IPFS
    results = {}
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            print(f"Uploading {file_name} to IPFS via Pinata...")
            
            # Upload the file to Pinata
            result = upload_to_pinata(file_path)
            
            if not result['success']:
                print(f"Error uploading {file_name}: {result['error']}")
                continue
            
            # Get the response data
            res = result['data']
            cid = res.get('IpfsHash')
            
            if not cid:
                print(f"Error: No CID returned for {file_name}")
                continue
            
            # Store the result
            results[file_name] = {
                'cid': cid,
                'size': res.get('PinSize', 0),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'gateway_url': f"https://ipfs.io/ipfs/{cid}",
                'pinata_url': f"https://gateway.pinata.cloud/ipfs/{cid}"
            }
            
            print(f"✅ Successfully uploaded {file_name}")
            print(f"   CID: {cid}")
            print(f"   Gateway URL: {results[file_name]['gateway_url']}")
            print(f"   Pinata Gateway: {results[file_name]['pinata_url']}")
            print("--------------------------------------------------")
            
            # Pin the file to ensure it remains available
            print(f"Pinning {cid}...")
            pin_result = pin_by_cid(cid, file_name)
            if pin_result['success']:
                print(f"✅ Successfully pinned {cid}")
            else:
                print(f"⚠️  Warning: Could not pin {cid}: {pin_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create an index file with all CIDs
    if create_index and results:
        try:
            index_data = {
                'files': results,
                'timestamp': datetime.now().isoformat(),
                'total_files': len(results)
            }
            
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            print(f"Created index file: {index_file}")
            
            # Upload the index file to IPFS
            print("Uploading index file to IPFS...")
            index_result = upload_to_pinata(index_file)
            
            if index_result['success']:
                res = index_result['data']
                index_cid = res.get('IpfsHash')
                
                print(f"✅ Successfully uploaded index file")
                print(f"   Index CID: {index_cid}")
                print(f"   Index URL: https://ipfs.io/ipfs/{index_cid}")
                
                # Add the index to the results
                results['_index'] = {
                    'cid': index_cid,
                    'size': res.get('PinSize', 0),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'gateway_url': f"https://ipfs.io/ipfs/{index_cid}"
                }
                
                # Pin the index file
                pin_by_cid(index_cid, "ipfs-index")
                
            else:
                print(f"Error uploading index file: {index_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Error creating index file: {e}")
    
    return results

def main():
    # Files to export to IPFS
    files_to_export = [
        'hack_analysis.json',
        'crypto_transactions.json',
        'images.json'
    ]
    
    # Filter out files that don't exist
    files_to_export = [f for f in files_to_export if os.path.exists(f)]
    
    if not files_to_export:
        print("Error: None of the specified files exist.")
        return
    
    print("Starting IPFS export process...")
    print(f"Found {len(files_to_export)} files to upload\n")
    
    # Export files to IPFS
    ipfs_results = export_to_ipfs(files_to_export)

    if ipfs_results:
        # Create a DataFrame for better visualization
        df_data = []
        for file_name, details in ipfs_results.items():
            if file_name != '_index':  # Skip the index entry
                df_data.append({
                    'File': file_name,
                    'CID': details['cid'],
                    'Size (bytes)': details['size'],
                    'Timestamp': details['timestamp'],
                    'Gateway URL': details['gateway_url']
                })
        
        if df_data:  # Only create DataFrame if we have data
            ipfs_df = pd.DataFrame(df_data)
            
            # Display the results
            print("\n" + "="*80)
            print("IPFS Export Summary:")
            print("="*80)
            print(ipfs_df.to_string(index=False))
            print("="*80)
            
            # Save results to a CSV file
            csv_file = 'ipfs_export_results.csv'
            ipfs_df.to_csv(csv_file, index=False)
            print(f"\nResults saved to: {csv_file}")
        
        print("\n✅ IPFS export process completed successfully!")
    else:
        print("\n❌ IPFS export process completed with errors.")

if __name__ == "__main__":
    main()

'''


# ## send files to sever

# In[41]:


import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB config from env
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]


# List of JSON files to load
json_files = [
    "hack_analysis.json",
    "crypto_transactions.json",
    "images.json"
]

for file_name in json_files:
    print(f"Processing {file_name}...")

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    document = {
        "source_file": file_name,
        "content": data,
        "updated_at": datetime.utcnow()
    }

    # Replace existing document or insert new one
    result = collection.replace_one(
        {"source_file": file_name},  # filter
        document,                   # replacement doc
        upsert=True                 # insert if not found
    )
    if result.matched_count:
        print(f"Updated existing document for {file_name}")
    else:
         print(f"Inserted new document for {file_name}")
        
     


# In[ ]:




