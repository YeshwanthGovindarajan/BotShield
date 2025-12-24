import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Files
users_file = 'subset_users.json'
tweets_file = 'tweet_subset.json'
selected_ids_file = 'selected_user_ids.json'
intermediate_file = 'user_tweets_intermediate.csv'

def main():
    print("Loading selected user IDs...")
    with open(selected_ids_file, 'r') as f:
        selected_ids_list = json.load(f)
        # Ensure IDs are integers
        selected_ids = set([int(uid) for uid in selected_ids_list])

    print(f"Found {len(selected_ids)} selected users.")

    print("Processing user profiles...")
    users_data = []
    with open(users_file, 'r') as f:
        # Load all users (assuming file is small enough as per analysis, ~72KB)
        all_users = json.load(f)
        
    for u in all_users:
        # Check if user is in selected subset
        # User ID in file matches format in selected_ids (int)
        uid = int(u['id'])
        if uid in selected_ids:
            # 1. Profile Completeness Score
            # Conditions:
            # - verified (+1)
            # - description exists (+1)
            # - description_length > 0 (+1)
            # - profile_image_url exists (+1)
            # - location exists (+1)
            
            score = 0
            # verified
            if u.get('verified') is True:
                score += 1
            
            # description
            desc = u.get('description')
            if desc is not None:
                score += 1 # Exists
                if len(desc) > 0:
                    score += 1 # Length > 0
            
            # profile_image_url
            if u.get('profile_image_url') is not None:
                score += 1
            
            # location
            if u.get('location') is not None:
                score += 1
            
            # 6. Follower / Following Ratio
            metrics = u.get('public_metrics', {})
            followers = metrics.get('followers_count', 0)
            following = metrics.get('following_count', 0)
            
            if following > 0:
                ratio = followers / following
            else:
                ratio = 0.0
            
            # Extract created_at for age calculation
            # Format: "2011-01-23 10:45:43+00:00"
            created_at_str = u.get('created_at')
            
            users_data.append({
                'user_id': uid,
                'profile_completeness_score': score,
                'follower_following_ratio': ratio,
                'user_created_at': created_at_str
            })

    df_users = pd.DataFrame(users_data)
    print(f"Processed {len(df_users)} user profiles.")

    print("Streaming tweets and mapping to users...")
    
    # We need to collect rows for intermediate CSV
    csv_rows = []
    
    # Helper to parse one line of json
    def parse_line(line):
        line = line.strip()
        # Remove trailing comma if present
        if line.endswith(','):
            line = line[:-1]
        # Ignore brackets
        if line == '[' or line == ']':
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    tweet_count = 0
    with open(tweets_file, 'r') as f:
        for line in f:
            tweet = parse_line(line)
            if tweet is None:
                continue
                
            author_id = int(tweet['author_id'])
            if author_id in selected_ids:
                # Extract required fields
                # user_id, tweet_id, tweet_text, created_at, like_count, reply_count
                metrics = tweet.get('public_metrics', {})
                
                like_count = metrics.get('like_count')
                if like_count is None: like_count = 0
                
                reply_count = metrics.get('reply_count')
                if reply_count is None: reply_count = 0
                
                row = {
                    'user_id': author_id,
                    'tweet_id': tweet['id'],
                    'tweet_text': tweet['text'],
                    'created_at': tweet['created_at'],
                    'like_count': like_count,
                    'reply_count': reply_count
                }
                csv_rows.append(row)
                tweet_count += 1

    print(f"Mapped {tweet_count} tweets from selected users.")
    
    # Convert to DataFrame
    df_tweets = pd.DataFrame(csv_rows)
    
    # Persist intermediate data (MANDATORY)
    df_tweets.to_csv(intermediate_file, index=False)
    print(f"Intermediate data saved to {intermediate_file}")
    
    # Computations using the intermediate DF
    print("Computing tweet-based features...")
    
    # Ensure datetime
    df_tweets['created_at_dt'] = pd.to_datetime(df_tweets['created_at'])
    
    # Determine reference date (max tweet date in dataset)
    # This represents 'now' for the static dataset
    if not df_tweets.empty:
        dataset_end_date = df_tweets['created_at_dt'].max()
    else:
        dataset_end_date = datetime.now(timezone.utc)
    
    # 2. Total likes received
    # 3. Total replies received
    # 5. Average tweet_text_length
    
    # Helper for text length
    df_tweets['text_len'] = df_tweets['tweet_text'].fillna("").apply(len)
    
    # GroupBy
    tweet_features = df_tweets.groupby('user_id').agg({
        'like_count': 'sum',
        'reply_count': 'sum',
        'text_len': 'mean',
        'tweet_id': 'count' # Count of tweets
    }).reset_index()
    
    tweet_features.rename(columns={
        'like_count': 'total_likes_received',
        'reply_count': 'total_replies_received',
        'text_len': 'avg_tweet_text_length',
        'tweet_id': 'obs_tweet_count'
    }, inplace=True)
    
    # Merge with users
    # "Do NOT drop users with missing data" -> Left join on users
    final_df = df_users.merge(tweet_features, on='user_id', how='left')
    
    # Fill NaN for users with no tweets
    fill_values = {
        'total_likes_received': 0,
        'total_replies_received': 0,
        'avg_tweet_text_length': 0,
        'obs_tweet_count': 0
    }
    final_df.fillna(value=fill_values, inplace=True)
    
    # 4. Average tweets per day
    # "total number of tweets / account_age_in_days"
    # Using observed tweets count from mapping
    
    # Calculate account age
    # user_created_at
    final_df['user_created_at_dt'] = pd.to_datetime(final_df['user_created_at'])
    
    # Age in days = (dataset_end_date - user_created_at).days
    final_df['account_age_days'] = (dataset_end_date - final_df['user_created_at_dt']).dt.days
    
    # Safety: age >= 1
    final_df['account_age_days'] = final_df['account_age_days'].clip(lower=1)
    
    final_df['avg_tweets_per_day'] = final_df['obs_tweet_count'] / final_df['account_age_days']
    
    # Final cleanup
    # Feature list ordered as requested
    feature_columns = [
        'profile_completeness_score',
        'total_likes_received',
        'total_replies_received',
        'avg_tweets_per_day',
        'avg_tweet_text_length',
        'follower_following_ratio'
    ]
    
    # Prepare Output
    output_df = final_df[['user_id'] + feature_columns].set_index('user_id')
    
    print("\n--- OUTPUT ---")
    print("Feature Names:")
    print(feature_columns)
    print("\nSample DataFrame Output (first 5):")
    print(output_df.head())
    
    # Also just output the dict form for inspection if needed, but DF is requested.
    # "Produce: 1. A DataFrame or dictionary ... 2. A list of feature names"
    
    # Printing the result DataFrame to stdout in execution might be messy if large, 
    # but I'll print head.
    
    # Save to file for verification
    output_df.to_csv('final_features.csv')
    print("\nFull feature table saved to 'final_features.csv'")

if __name__ == "__main__":
    main()
