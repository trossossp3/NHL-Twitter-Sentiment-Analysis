import requests
import os
import json
import pandas as pd
import csv
# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = "AAAAAAAAAAAAAAAAAAAAANasXgEAAAAAGxSlvLUS7hGWSmi8Vqy%2Bd19ahe0%3DOwQcm0mMYQGOGZQ80m2VbgfVdKNhri1me8snB03devHdSjvbi1"


search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
start_time = '2022-01-06T02:20:00Z'
end_time = '2022-01-06T05:30:00Z'
query_params = {'query': '#leafsforever -is:retweet -is:reply','tweet.fields': 'author_id,created_at', 'max_results': 100, 'start_time': start_time, 'end_time': end_time}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.url)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    json_response = connect_to_endpoint(search_url, query_params)
    # print(json.dumps(json_response, indent=4, sort_keys=True))
    # with open('data.json', 'w') as f:
    #     json.dump(json_response, f)
    # print(json_response['data'])
    to_csv(json_response,"data/tweets_raw.csv")
    

def to_csv(json_response, out):
    csvFile = open(out, "w", newline=None, encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    for tweet in json_response['data']:
        text = (tweet['text'])
        time = tweet['created_at']
        csvWriter.writerow([text])
        # csvWriter.writerow(["*"*100])
    csvFile.close()




if __name__ == "__main__":
    main()