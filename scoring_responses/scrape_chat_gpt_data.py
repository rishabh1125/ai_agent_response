from langchain_community.document_loaders import AsyncHtmlLoader
import re
from pprint import pprint
import json

def find_text_format(text):
  """
  Finds all occurrences of text matching a specific format using regex.

  Args:
      text: The input text to search.

  Returns:
      list: A list of all matches found in the text.
  """
  # Define the regular expression pattern
  #pattern = r"{\"content_type\":\\\"text\\\",\"parts\\\":[\\\"{\n  \\\"user_actions\\\": 0,\n  \\\"agent_actions\\\": 0,\n  \\\"agent_action_attributes\\\": 0,\n  \\\"format\": 1,\n  \\\"tone\": 1\n}\"]}"
  pattern = r"user_actions"
  # Find all occurrences using re.findall
  indices = [match.start() for match in re.finditer(r"user_actions", text)]
  return indices

url_list = ["https://chat.openai.com/share/27e95eab-76d2-45f6-b198-ba3a9501f8da"]
loader = AsyncHtmlLoader(url_list)
docs = loader.load()
web_text = docs[0].page_content
matches_index = find_text_format(web_text)


values_json = []
if matches_index!=[]:
  for i in matches_index[1:]:
    if web_text[i+16] in '01':
      json_string = '{' + web_text[i-2:i+114]
      json_string = json_string.replace('\\n','')
      json_string = json_string.replace('\\','')
      json_object = json.loads(json_string)
      values_json.append([json_object['user_actions'],json_object['agent_actions'],json_object['agent_action_attributes'],json_object['format'],json_object['tone']])
  json.dump(values_json,open('training_data/y_train.json','w+'))
else:
  print("No matches found.")