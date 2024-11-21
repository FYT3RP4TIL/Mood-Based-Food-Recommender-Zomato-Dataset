# 🍽️ Mood-Driven Food Recommender - Zomato Dataset 

## 🎯 Project Overview
This data science project explores the fascinating relationship between human emotions and food choices using comprehensive datasets from Zomato (restaurant data) and a food choices survey.

## 🛠️ Technologies Used
- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn
- 📈 Matplotlib
- 🌈 Seaborn
  
### 🌟 Key Features
- 📊 Analyze restaurant cuisine trends in New Delhi
- 🗺️ Map restaurant ratings across different city clusters
- 😋 Identify comfort foods for various emotional states
- 🍲 Provide restaurant recommendations based on mood

## 📊 Datasets Used
1. **Zomato Restaurants Dataset**
   - 📍 Location: New Delhi, India
   - 🏘️ Contains details about 3,975 restaurants
   - 📋 Includes information like cuisine types, ratings, and geolocation

2. **Food Choices Survey Dataset**
   - 👥 125 survey respondents
   - 📝 61 variables capturing food preferences, emotional states, and eating habits

### What Are the Most Famous Cuisines in Delhi ?
```python
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Cnt_cuisine = Counter()

res_data['Cuisines'].str.split(',').apply(lambda cuisines: Cnt_cuisine.update([c.strip() for c in cuisines]))

cnt = pd.DataFrame.from_dict(Cnt_cuisine, orient='index', columns=['cnt'])
cnt.sort_values('cnt', ascending=False, inplace=True)

tmp_cnt = cnt.head(10)

print(tmp_cnt)
print()

with plt.style.context('bmh'):
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    sns.barplot(
        x=tmp_cnt.index,
        y='cnt',
        data=tmp_cnt,
        ax=ax1,
        palette=sns.color_palette('Blues_d', 10),
        hue=tmp_cnt.index,
        dodge=False,
        legend=False
    )
    ax1.set_title('# Cuisine')
    ax1.tick_params(axis='x', rotation=70)

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    sns.countplot(
        x=res_data['fusion_num'],
        ax=ax2,
        palette=sns.color_palette('Blues_d', res_data['fusion_num'].nunique()),
        hue=res_data['fusion_num'],
        dodge=False,
        legend=False
    )
    ax2.set_title('# Cuisine Provided')
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.show()

print()
print('# Unique Cuisine:', len(Cnt_cuisine))

```
```
Top 10         cnt
North Indian  1791
Chinese       1268
Fast Food     1001
Mughlai        485
Italian        355
Continental    340
Bakery         308
South Indian   306
Desserts       300
Cafe           290

# Unique Cuisine: 78
```
![download (1)](https://github.com/user-attachments/assets/995e96a5-166e-4b60-8743-1f26ed41c820)

## 🔍 Key Functions

### 🍽️ Top Comfort Foods Finder
```python
def find_top_comfort_foods(mood, top_n=10):
    """
    Find top comfort foods for a specific mood
    
    Args:
        mood (str): Emotional state to search
        top_n (int): Number of top foods to return
    
    Returns:
        List of top comfort foods
    """
    top_comfort_foods = search_comfort(mood)
    return top_comfort_foods[:top_n]
```
### 🗺️ K-Means Restaurant Clustering
```python
def cluster_restaurant_ratings(data, n_clusters=7):
    """
    Cluster restaurants based on location and ratings
    
    Args:
        data (DataFrame): Restaurant dataset
        n_clusters (int): Number of geographical clusters
    
    Returns:
        DataFrame with cluster information and median ratings
    """
    kmeans = KMeans(n_clusters=n_clusters).fit(data[['Longitude', 'Latitude']])
    data['cluster'] = kmeans.labels_
    
    cluster_stats = data.groupby('cluster')[['Longitude', 'Latitude', 'Aggregate rating']].agg({
        'Longitude': 'mean',
        'Latitude': 'mean',
        'Aggregate rating': 'median'
    }).reset_index()
    
    return cluster_stats
```
![download](https://github.com/user-attachments/assets/83077afc-3b30-4cb7-96e3-70594815f97d)

## 🍦 Mood-Based Comfort Food Insights

### 😢 Sad Mood Top 3 Comfort Foods:
1. Ice Cream
2. Pizza
3. Chips

### 😄 Happy Mood Top 3 Comfort Foods:
1. Pizza
2. Ice Cream
3. Chicken Wings

### 🍽️ Hunger Top 3 Comfort Foods:
1. Mac and Cheese
2. Burger
3. Ice Cream

## 🍽️ Restaurant Recommendation Based on Happy Mood

### Code Explanation

The code snippet demonstrates how to find top restaurants serving comfort foods associated with a "happy" mood:

```python
def get_happy_mood_restaurants(res_data):
    """
    Find top-rated restaurants serving comfort foods for a happy mood
    
    Args:
    res_data (DataFrame): Restaurant dataset containing cuisines and ratings
    
    Returns:
    DataFrame: Top 3 highest-rated restaurants serving happy mood comfort foods
    """
    # Identify top comfort foods for happy mood
    happy_foods = ['pizza', 'ice cream', 'chicken wings']
    
    # Filter restaurants based on happy mood comfort foods
    happy_restaurants = res_data[
        res_data['Cuisines'].str.contains('|'.join(happy_foods), case=False)
    ].sort_values(by='Aggregate rating', ascending=False).head(3)
    
    return happy_restaurants

# Example usage
happy_restaurants = get_happy_mood_restaurants(res_data)
print(happy_restaurants[['Restaurant Name', 'Cuisines', 'Aggregate rating']])
```

### Insights from Results

1. **Owl is Well** 
   - Rating: 4.5/5
   - Cuisines: Burger, American, Fast Food, Italian, Pizza
   - Location: Greater Kailash (GK) 1, New Delhi

2. **Civil House**
   - Rating: 4.2/5
   - Cuisines: European, Continental, Pizza
   - Location: Khan Market, New Delhi

3. **Tossin Pizza**
   - Rating: 4.1/5
   - Cuisines: Pizza, Italian
   - Location: Safdarjung Enclave, New Delhi

### Mood-Food Correlation Methodology

1. **Data Collection**: 
   - Survey of 125 respondents
   - Collected data on food preferences and emotional states

2. **Comfort Food Identification**:
   - Analyzed correlation between emotions and food choices
   - Created mapping of comfort foods for different moods

3. **Restaurant Recommendation Algorithm**:
   - Match comfort foods with restaurant cuisines
   - Rank restaurants based on aggregate ratings
   - Provide top recommendations for each mood

### Potential Future Improvements
- Implement machine learning model for more personalized recommendations
- Include more granular mood categories
- Add user preference learning mechanism
- Integrate real-time restaurant availability
