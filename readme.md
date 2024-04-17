<h2>AI response scoring model</h2>

<h4>Objective</h4>
We are building model that gives reliable answer to gpt4 based prompts, based on similarity.

<h4>Files directory</h4>
<ul>
<li><b>Data/:</b> This stores data cleaned data of coontext, human_response and AI response. 
'cleaned_equal.csv' has reponses where AI and human reponse match.
'cleaned_non_equal.csv' has reponses where AI and human reponse doesn't match. </li>
<li><b>preview_data</b>: This is just to view data in text format rather than dataframe.</li>
<li><b>preview_data/</b>: This is just to view data in text format rather than dataframe.</li>
<li><b>training_data/</b>: This folder stores data X_train(tokenised format of context/response/agent_response) and y_train: LLM generated score against them.</li>
<li><b>scoring_responses/</b>: This directory has files to evaluate and score AI responses - </li>
    <ul>
        <li>format_prompts.py: Python file to take random data and generate prompt (to be used in ChatGPT) for the same. </li>
        <li>prompts/: saves prompts used for GPT3.5 to generate scores for unlabeled data. </li>
        <li>scrape_chat_gpt_data.py: Python file to scrape data from ChatGPT chat, saves to 'training_data/'. </li>
        <li>training_scoring_model.py: This file uses data to train our scoring model and saves to 'model/'.</li>
        <li>scoring_responses.py: This file is used to generate scores for responses.</li>

    </ul>
</ul>