import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import nltk
from nltk.corpus import stopwords
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy.stats as stats
from scipy.stats import normaltest, shapiro, mannwhitneyu, zscore, f_oneway


def my_read_csv(nrows = None):
    #My custom read csv for this steam reviews dataset
    if nrows == None:
        print("Getting all the rows")
        df = pd.read_csv('steam_reviews.csv', on_bad_lines='skip', index_col=0)
    else:
        df = pd.read_csv('steam_reviews.csv', nrows=nrows, on_bad_lines='skip', index_col=0)

    df = df.drop_duplicates(subset=['review_id'])
    # Convert columns from Unix time (seconds) to datetime
    df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s', errors='coerce')
    df['timestamp_updated'] = pd.to_datetime(df['timestamp_updated'], unit='s', errors='coerce')
    df['author.last_played'] = pd.to_datetime(df['author.last_played'], unit='s', errors='coerce')

    #Convert to strings where necessary
    df['author.steamid'] = df['author.steamid'].astype('string')
    df['review_id'] = df['review_id'].astype('string')
    df['app_id'] = df['app_id'].astype('string')
    return df

#helper function
def cv(x):
    if np.mean(x)==0:
        return 0
    else:
        return np.std(x, ddof=1) / np.mean(x) * 100

#My describe function to show more neatly the dataframe
def my_describe(df):
    descr_stat = df.describe().T
    numeric_cols = df.select_dtypes(include=[np.number])
    descr_stat['median'] = numeric_cols.median()
    descr_stat['cv'] = numeric_cols.apply(cv)
    descr_stat['range'] = numeric_cols.max() - numeric_cols.min()
    descr_stat_df=pd.DataFrame(descr_stat)
    return descr_stat_df


def show_game_and_language_percentage(df):
    game_counts = df['app_name'].value_counts(normalize=True) * 100
    language_counts = df['language'].value_counts(normalize=True) * 100

    # Function to group small categories into 'Other'
    def group_small_categories(counts, threshold=1.5):
        small_categories = counts[counts <= threshold].sum()
        counts = counts[counts >= threshold]
        counts['Other'] = small_categories
        return counts

    # Group games and languages with less than 5% into 'Other'
    game_counts_grouped = group_small_categories(game_counts)
    language_counts_grouped = group_small_categories(language_counts)

    # Create subplots for both Game and Language pie charts
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Pie chart for Games
    ax[0].pie(game_counts_grouped, labels=game_counts_grouped.index, autopct='%1.1f%%', startangle=90)
    ax[0].set_title('Game Distribution')

    # Pie chart for Languages
    ax[1].pie(language_counts_grouped, labels=language_counts_grouped.index, autopct='%1.1f%%', startangle=90)
    ax[1].set_title('Language Distribution')

    # Show the charts
    plt.tight_layout()
    plt.show()


def show_monthly_trends(df):
    #group months
    df['month'] = df['timestamp_created'].dt.strftime('%B')
    groupbyMonth = df.groupby('month')
    monthly_count = groupbyMonth.size()

    #x labels
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts_sorted = monthly_count.reindex(month_order)

    #plot the figure
    plt.figure(figsize=(12, 8))
    plt.plot(monthly_counts_sorted.index, monthly_counts_sorted.values)
    plt.title('Number of Reviews by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def show_daily_trends(df):
    df['minute_hour'] = df['timestamp_created'].dt.strftime('%H:%M')

    # Count reviews by half-hour
    minute_hour_counts = df['minute_hour'].value_counts().sort_index()
    # Filter the index to show only half-hour intervals ('00', '30' minutes)
    half_hour_indices = [i for i, time in enumerate(minute_hour_counts.index) if time[-2:] in ['00', '30']]
    smoothed_values = gaussian_filter1d(minute_hour_counts.values, sigma=5)


    plt.figure(figsize=(15, 6))
    plt.plot(minute_hour_counts.index, minute_hour_counts.values, marker='', linestyle='-')
    plt.plot(minute_hour_counts.index, smoothed_values, color='red', label='Smoothed line') #a smoothed line using a Gaussian filter
    plt.title('Number of Reviews')
    plt.xlabel('Time (Hour:Minute)')
    plt.ylabel('Number of Reviews')

    # Display only every half hour in x-ticks
    plt.xticks(ticks=half_hour_indices, labels=np.array(minute_hour_counts.index)[half_hour_indices], rotation=45)

    # Remove MaxNLocator and MultipleLocator since we are manually controlling ticks
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plotTimeInterval(df, start, end):
    # Create a new column for formatted time
    df['minute_hour'] = df['timestamp_created'].dt.strftime('%H:%M')
    
    # Count occurrences and filter by time range
    minute_hour_counts = df['minute_hour'].value_counts().sort_index()
    minute_hour_counts = minute_hour_counts[(minute_hour_counts.index >= start) & (minute_hour_counts.index <= end)]

    # Apply Gaussian smoothing
    smoothed_values = gaussian_filter1d(minute_hour_counts.values, sigma=3)

    plt.figure(figsize=(15, 6))
    # Add labels to the plots
    plt.plot(minute_hour_counts.index, minute_hour_counts.values, marker='', linestyle='-', label='Original Data')
    plt.plot(minute_hour_counts.index, smoothed_values, color='red', label='Smoothed Line')

    plt.title('Number of Reviews')
    plt.xlabel('Time (Hour:Minute)')
    plt.ylabel('Number of Reviews')

    # Create x-ticks for every 5 minutes within the start and end range
    time_labels = pd.date_range(start=start, end=end, freq='5min').strftime('%H:%M')

    # Determine tick positions for 5-minute intervals
    tick_positions = np.arange(0, len(minute_hour_counts), 5)  # Every 5th index

    # Set the x-ticks and labels
    plt.xticks(ticks=tick_positions, labels=time_labels[:len(tick_positions)], rotation=45)

    plt.legend()  # Ensure the legend is shown correctly
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def min_and_max_game_reviews(df):
    # Number of reviews received by each application: count of the review IDs of the reviews it received
    grouped_apps = df.groupby('app_name').review_id.count()
    print(f"'{grouped_apps.idxmin()}' was reviewed least times, it was reviewed {grouped_apps.min()} times.")
    print(f"'{grouped_apps.idxmax()}' was reviewed most times, it was reviewed {grouped_apps.max()} times.")


def review_per_game(df):
    # Change of scale of the number of reviews in Millions to make the graph better and sorting the values to plot in descending order 
    reviews_per_app = ((df['app_name'].value_counts())/1000000).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    reviews_per_app[0:50].plot.bar(ylabel = 'Number of reviews', xlabel = 'App name', title = 'Number of reviews per application (in Millions)')


def top5_reviews_analysis(df):
    grouped_apps = df.groupby('app_name').review_id.count()
    # Find the top 5 applications by number of reviews
    top = grouped_apps.sort_values(ascending = False)[0:5]

    # Extract the part of the dataset relative to the top 5 apps
    top_5_df = df[df['app_name'].isin(top.index)]

    # For each application, find how many reviews came from people who purchased the application and how many from users that received the app for free
    purchased = top_5_df.groupby(['app_name', 'steam_purchase'])['review_id'].count().unstack()
    purchased.columns = ['Received for Free', 'Purchased']

    # Initialize the plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()

    for app, ax in zip(purchased.index, axs):
        # Compute the percentage of people that purchased the app
        purchased_count = purchased.loc[app, 'Purchased']
        free_count = purchased.loc[app, 'Received for Free']
        purchased_percentage = (purchased_count/(purchased_count+free_count))*100
        print(f"{purchased_count} users purchased '{app}', it corresponds to {round(purchased_percentage, 1)}% of its total users. \n{round(100-purchased_percentage, 1)}% of the users received '{app}' for free, that is {free_count} users.\n")

        # Pie plot
        labels = ['Purchased', 'Received for Free']
        sizes = [purchased_count, free_count]
        colors = ['#ff5733', '#3380ff']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(f"Number of reviews\n from people that purchased \n '{app}' \nand from those that received for free.")
        ax.axis('equal')

    # Remove unused spot in the bottom-left corner of the graph
    axs[-1].axis('off')


    plt.tight_layout()
    plt.show()

def min_max_game_reccomendations(df):
    # Number of recommendations received by each application: sum of the recommended == True the app received
    grouped_recommendations = df.groupby('app_name').recommended.sum()
    print(f"'{grouped_recommendations.idxmin()}' was recommended least times, it was recommended {grouped_recommendations.min()} times.")
    print(f"'{grouped_recommendations.idxmax()}' was recommended most times, it was recommended {grouped_recommendations.max()} times.")

def correlation_reccomendations(df):
    # Extract from the dataset the columns that contain the name of the application, if the review recommended it and the review id
    df_scores = df[['app_name', 'recommended', 'review_id']]
    recommended_group = df_scores.groupby(['app_name', 'recommended']).review_id.count().unstack()
    recommended_group.columns = ['Not recommended', 'Recommended']
    new_col = {}


    for app in recommended_group.index:
        rec = recommended_group.loc[app, 'Recommended']
        # Compute and append the review score as the ratio of times that the app was recommended over the total number of reviews it received
        review_score = rec/(rec+recommended_group.loc[app, 'Not recommended'])
        new_col[app] = review_score

    recommended_group['review_score'] = new_col

    recommended_group = pd.DataFrame(recommended_group)

    # Check the normality of the distributions of the number of recommendations and the review score using the Shapiro-Wilk test
    W_rec, normality_recommendations = stats.shapiro(recommended_group['Recommended'])
    W_rev, normality_review_score = stats.shapiro(recommended_group['review_score'])
    print(f'The p-value associated with the Shapiro-Wilk test to verify that the distribution of the number of recommendations is normal is equal to {normality_recommendations}, with associated value {W_rec}.')
    print(f'The p-value associated with the Shapiro-Wilk test to verify that the distribution of the review score is normal is equal to {normality_review_score}, with associated value {W_rev}.\n')


    # Spearman
    coeff_spear, p_spear = stats.spearmanr(recommended_group['Recommended'], recommended_group['review_score'])
    print(f"The Spearman test coefficient between the number of recommendations and the applications' review scores is equal to {coeff_spear}.")
    print(f"The p-value associated to the hypothesis test to verify that the two variables are correlated is equal to {p_spear}.")


    # Scatter plot of the number of recommendations and review score
    plt.figure(figsize=(12, 8))
    plt.scatter(recommended_group['Recommended']/100000, recommended_group['review_score'], alpha=0.8)
    plt.xlabel('Number of recommendations (in Hundreds of Thousands)')
    plt.ylabel('Review score')
    plt.title("Scatterplot of the Number of Recommendations vs Review Score")
    plt.show()


def my_game_scores(df):
    #My version of calculating the game score GM = RECOMMENDED / TOT_REVIEW
    df_game_scores=df.groupby(['app_name','recommended']).review_id.count().unstack()
    df_game_scores['total_reviews'] = df_game_scores.sum(axis=1)
    df_game_scores = df_game_scores.fillna(0)
    df_game_scores['review_score_games'] = df_game_scores[True] / df_game_scores['total_reviews']
    df_game_scores = df_game_scores.reset_index('app_name')

    return df_game_scores

def score_by_playtime(df):
    #Function to see the score by playtime
    df_game_scores = my_game_scores(df)
    scores = df_game_scores[['app_name','review_score_games']]
    scores = scores.reset_index()
    scores = scores.set_index('app_name')  

    df_score_playtime=df[['app_name','author.playtime_forever']].copy()
    df_score_playtime['review_score_games']=df_score_playtime['app_name'].map(scores['review_score_games'])

    return df_score_playtime

def show_score_by_playtime(df_score_playtime):
    plt.figure(figsize=(12, 8))
    plt.scatter(df_score_playtime['author.playtime_forever'],df_score_playtime['review_score_games'])
    plt.title('Scatter plot')
    plt.xlabel('author.playtime_forever')
    plt.ylabel('review_score_games')
    plt.show()


def score_by_playtime_stat(df_score_playtime):
    groups = [group['author.playtime_forever'].values for name, group in df_score_playtime.groupby('review_score_games')]
    f_statistic, p_value = f_oneway(*groups)
    print(f'F-statistic: {f_statistic}')
    print(f'P-value: {p_value}')

    alpha = 0.05 
    if p_value < alpha:
        print("There are significant differences between the means of the categories of review_score_games.")
    else:
        print("There are no significant differences between the means of the categories of review_score_games.")


def author_scores(df):
    df_author_scores=df.groupby(['author.steamid','recommended']).review_id.count().unstack()
    df_author_scores[True] = df_author_scores[True].fillna(0)
    df_author_scores[False] = df_author_scores[False].fillna(0)

    df_author_scores['total_reviews'] = df_author_scores.sum(axis=1)
    df_author_scores['review_score_author'] = df_author_scores[True] / df_author_scores['total_reviews']
    df_author_scores = df_author_scores.reset_index('author.steamid')

    return df_author_scores

def score_by_playtime2(df, df_author_scores):
    scores2 = df_author_scores[['author.steamid','review_score_author']]
    scores2 = scores2.reset_index()
    scores2 = scores2.set_index('author.steamid')  

    df_score_playtime2=df[['author.steamid','author.playtime_forever']].copy()

    df_score_playtime2['review_score_author']=df_score_playtime2['author.steamid'].map(scores2['review_score_author'])

    return df_score_playtime2


def show_score_by_playtime2(df_score_playtime2):
    plt.figure(figsize=(12, 8))
    plt.scatter(df_score_playtime2['author.playtime_forever'],df_score_playtime2['review_score_author'])
    plt.title('Scatter plot')
    plt.xlabel('author.playtime_forever')
    plt.ylabel('review_score_games')
    plt.show()

def stat_author_scores(df_score_playtime2):
    groups_2 = [group['author.playtime_forever'].values for name, group in df_score_playtime2.groupby('review_score_author')]
    f_statistic2, p_value2 = f_oneway(*groups_2)
    print(f'F-statistic: {f_statistic2}')
    print(f'P-value: {p_value2}')
    alpha = 0.05 
    if p_value2 < alpha:
        print("There are significant differences between the means of the categories of review_score_author.")
    else:
        print("There are no significant differences between the means of the categories of review_score_author.")


def filter_greater_than_quantile(df, q):
    # Calculate the qth percentile of playtime for each game
    perc_time = df.groupby('app_name')['author.playtime_forever'].quantile(q)
    perc_time = perc_time.reset_index()  # Reset the index for easier handling
    perc_time.columns = ['app_name', f"{int(q * 100)}_perc"]  # Rename columns
    perc_time = perc_time.set_index('app_name')  # Set 'app_name' as the index

    # Create a new DataFrame to compare playtime with the qth percentile
    df_time = df[['app_name', 'author.playtime_forever']].copy()
    df_time[f"{int(q * 100)}_perc"] = df_time['app_name'].map(perc_time[f"{int(q * 100)}_perc"])

    # Check if the playtime is greater than or equal to the qth percentile
    result = df['author.playtime_forever'] >= df_time[f"{int(q * 100)}_perc"]
    return result

def filter_less_than_quantile(df, q):
    # Calculate the qth percentile of playtime for each game
    perc_time = df.groupby('app_name')['author.playtime_forever'].quantile(q)
    perc_time = perc_time.reset_index()  # Reset the index for easier handling
    perc_time.columns = ['app_name', f"{int(q * 100)}_perc"]  # Rename columns
    perc_time = perc_time.set_index('app_name')  # Set 'app_name' as the index

    # Create a new DataFrame to compare playtime with the qth percentile
    df_time = df[['app_name', 'author.playtime_forever']].copy()
    df_time[f"{int(q * 100)}_perc"] = df_time['app_name'].map(perc_time[f"{int(q * 100)}_perc"])

    # Check if the playtime is less than or equal to the qth percentile
    result = df['author.playtime_forever'] <= df_time[f"{int(q * 100)}_perc"]
    return result




def comparison_game_author_score(df, df_score_playtime, df_score_playtime2, veteran_q=0.8, newbie_q=0.5):
    # Filter data based on the veteran percentile condition
    result_veteran = filter_greater_than_quantile(df, veteran_q)
    result_veteran_author = df_score_playtime2[result_veteran]
    result_veteran_game = df_score_playtime[result_veteran]

    # Calculate mean, median, and standard deviation for author scores above the veteran percentile
    mean_veteran_author = result_veteran_author['review_score_author'].mean()
    mean_veteran_game = result_veteran_game['review_score_games'].mean()
    median_veteran_author = result_veteran_author['review_score_author'].median()
    median_veteran_game = result_veteran_game['review_score_games'].median()
    std_veteran_author = result_veteran_author['review_score_author'].std()
    std_veteran_game = result_veteran_game['review_score_games'].std()

    # Filter data based on the newbie percentile condition
    result_newbie = filter_less_than_quantile(df, newbie_q)
    result_newbie_author = df_score_playtime2[result_newbie]
    result_newbie_game = df_score_playtime[result_newbie]

    # Calculate mean, median, and standard deviation for author scores below the newbie percentile
    mean_newbie_author = result_newbie_author['review_score_author'].mean()
    mean_newbie_game = result_newbie_game['review_score_games'].mean()
    median_newbie_author = result_newbie_author['review_score_author'].median()
    median_newbie_game = result_newbie_game['review_score_games'].median()
    std_newbie_author = result_newbie_author['review_score_author'].std()
    std_newbie_game = result_newbie_game['review_score_games'].std()

    # Create DataFrames to compare the statistics of author scores
    comparison_author = pd.DataFrame(
        [
            [mean_veteran_author, median_veteran_author, std_veteran_author],
            [mean_newbie_author, median_newbie_author, std_newbie_author]
        ],
        columns=['Mean', 'Median', 'Standard Deviation'],
        index=[f'Over {int(veteran_q * 100)}% Author_score (Veteran)', f'Under {int(newbie_q * 100)}% Author_score (Newbie)']
    )
    
    # Create DataFrames to compare the statistics of game scores
    comparison_game = pd.DataFrame(
        [
            [mean_veteran_game, median_veteran_game, std_veteran_game],
            [mean_newbie_game, median_newbie_game, std_newbie_game]
        ],
        columns=['Mean', 'Median', 'Standard Deviation'],
        index=[f'Over {int(veteran_q * 100)}% Game_score (Veteran)', f'Under {int(newbie_q * 100)}% Game_score (Newbie)']
    )

    # Return the two comparison DataFrames
    return comparison_author, comparison_game


def top10_reviewers(df):
    top_reviewers = df.groupby('author.steamid')['review_id'].count().sort_values(ascending=False)[0:10]
    # Extract the part of the dataset relative to the top 10 reviewers
    df_reviewers = df[df['author.steamid'].isin(top_reviewers.index)]
    author_grouped = df_reviewers.groupby(['author.steamid', 'language'])['review_id'].count().unstack()


    # Pie chart
    sizes = [author_grouped[lang].sum() for lang in author_grouped.columns]
    plt.figure(figsize=(5,5))
    cmap = plt.get_cmap('Paired') 
    colors = cmap(np.linspace(0, 1, len(sizes)))
    plt.pie(sizes, labels=author_grouped.columns, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Percentage of each language\n used by the top 10 reviewers')


def reviews_changed(english, spanish):
    english_change = english[english['timestamp_created'] != english['timestamp_updated']]
    spanish_change = spanish[spanish['timestamp_created'] != spanish['timestamp_updated']]
    english_count = len(english)
    spanish_count = len(spanish)

    changed_english_proportion = len(english_change) / english_count
    changed_spanish_proportion = len(spanish_change) / spanish_count

    # Step 4: Create the pie charts
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # English Reviews Pie Chart
    axs[0].pie([changed_english_proportion, 1 - changed_english_proportion], 
                labels=['YES', 'NO'], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['blue','skyblue'])
    axs[0].set_title('English Reviews Changed')

    # Spanish Reviews Pie Chart
    axs[1].pie([changed_spanish_proportion, 1 - changed_spanish_proportion], 
                labels=['YES', 'NO'], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['red','lightcoral'])
    axs[1].set_title('Spanish Reviews Changed')

    # Equal aspect ratio ensures that pie charts are circles
    for ax in axs:
        ax.axis('equal')

    plt.tight_layout()
    plt.show()

def average_game_and_review(english,spanish):
    english_unique = english.drop_duplicates(subset = 'author.steamid')
    spanish_unique = spanish.drop_duplicates(subset = 'author.steamid')

    #I need to cut the outliners because some crazy people have 12 millions games
    english_95_quantile = english_unique['author.num_games_owned'].quantile(0.95)
    spanish_95_quantile = spanish_unique['author.num_games_owned'].quantile(0.95)
    english_mean_games = english_unique[english_unique['author.num_games_owned'] < english_95_quantile]['author.num_games_owned'].mean()
    spanish_mean_games = spanish_unique[spanish_unique['author.num_games_owned'] < spanish_95_quantile]['author.num_games_owned'].mean()


    english_mean_reviews = english_unique[english_unique['author.num_reviews'] < english_95_quantile]['author.num_reviews'].mean()
    spanish_mean_reviews = spanish_unique[spanish_unique['author.num_reviews'] < spanish_95_quantile]['author.num_reviews'].mean()

    english_mean_reviews_per_game = english_mean_reviews / english_mean_games * 100
    spanish_mean_reviews_per_game = spanish_mean_reviews / spanish_mean_games * 100
    fig , axs = plt.subplots(1,3, figsize = (12,6))

    # First bar plot for average games Owned
    axs[0].bar(['English', 'Spanish'], [english_mean_games, spanish_mean_games], color=['blue', 'red'])
    axs[0].set_title('Average Games Owned by Each Group')
    axs[0].set_xlabel('Language Group')
    axs[0].set_ylabel('Average Number of Games')

    # Second bar plot for average reviews
    axs[1].bar(['English', 'Spanish'], [english_mean_reviews, spanish_mean_reviews], color=['blue', 'red'])
    axs[1].set_title('Average Reviews by Each Group')
    axs[1].set_xlabel('Language Group')
    axs[1].set_ylabel('Average Number of Reviews')

    # Third bar plot for proportion of reviews per game
    axs[2].bar(['English', 'Spanish'], [english_mean_reviews_per_game, spanish_mean_reviews_per_game], color=['blue', 'red'])
    axs[2].set_title('Proportionate Reviews by Each Group')
    axs[2].set_xlabel('Language Group')
    axs[2].set_ylabel('Proportion of Reviews per game')

    # Show the plots
    plt.tight_layout()                             
    plt.show()