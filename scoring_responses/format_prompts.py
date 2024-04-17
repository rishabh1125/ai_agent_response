import pandas as pd
import json
from random import randint

def dataframe_to_json_list(df,ids):
    """
    Converts a DataFrame to a list of JSON objects.

    Args:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    list: List of JSON objects.
    """
    json_list = []
    index_list = []
    for id in ids:
        json_list.append({
            'prev_context':df['prev_context'][id],
            'response':df['response'][id],
            'agent_response':df['agent_response'][id]
        }) 
        index_list.append(id)
    return index_list,json_list

# Example usage:
# Create a sample DataFrame
df = pd.read_csv('../../Data/cleaned_non_equal.csv')
ids = [randint(0,len(df)) for i in range(98)]
# Convert DataFrame to list of JSON objects
index_list,json_list = dataframe_to_json_list(df,ids)


base_prompt = """Given below are context, human response and AI agent response to an e-commerce. You have to score  similarity between the 2 resposnes based on follwing criteria, please give 1 for same/ 0 mark for different values.
- User action needed (eg. Share order details, Share image of product, etc)
- Agent actions updated(eg. Refund has been processed,  Your address is updated etc., )
- Agent action attribute(eg. Refund amount processed , Days to reflect changes etc) 
- Format( should salutation, body,regards in order) 
- Tone (values: apologetic/gratitude, etc)
Share your answer as json object of format {"user_actions": ,"agent_actions": , "agnet_action_attributes": ,"format":,"tone":,""} - 


"""
for id,json_item in enumerate(json_list):
    prompt = base_prompt + json.dumps(json_item)
    open(f'data/prompt_{id}.txt','+w').write(prompt)
json.dump(index_list,open('data/indexes.txt','+w'))
