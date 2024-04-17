from train_scoring_model import ANNModel
from transformers import BertTokenizer
import torch
import pandas as pd
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = ANNModel()

# Load the saved model parameters
model.load_state_dict(torch.load('model/torch_ann_model.pth'))

def scoring_response(context,response,agent_repsonse):
    """
    This scoring mechanism will use ANN model to score tests.
    """
    context_tokens = tokenizer.encode(context, add_special_tokens=True, max_length=1024, pad_to_max_length=True, truncation=True)
    response_token = tokenizer.encode(response, add_special_tokens=True, max_length=512, pad_to_max_length=True, truncation=True)
    agent_response_token = tokenizer.encode(agent_repsonse, add_special_tokens=True, max_length=512, pad_to_max_length=True, truncation=True)
    _ = context_tokens
    _.extend(response_token)
    _.extend(agent_response_token)
    # Perform inference with no gradient computation
    with torch.no_grad():
        # Your inference code here
        # Example:
        input_data = torch.tensor(_).float()  # Example input data
        output = model(input_data)

    # Output
    print("Score:", output)

if __name__=="__main__":
    scoring_response("""Customer's Message: New customer message on <DATE_TIME> at <DATE_TIME>

You received a new message from your online store's contact
   form.
Country Code:
<LOCATION>
Name:
<PERSON>
Email:
<EMAIL_ADDRESS>
Phone:
<UK_NHS>
Body:
having issues with both sets of hearing aids i bought.  Bought one set for my sister (the pro) and one set for me the <LOCATION>.
   on mine they will not charge.  On <PERSON>'s the one would not work.Would like to return both sets and possibly try again?
   Please give me instructions on how to return as there were none in the package.  Thank you.

   <PERSON>
   3 Deer Run Drive
   Wheeling, WV 26003""", """Hello <PERSON>,

We're sorry to hear about the issues you're experiencing with your hearing aids. We have sent a return label to your email address <EMAIL_ADDRESS> and have initiated an exchange for both sets of hearing aids. Please check your inbox for instructions.

Best regards,
<PERSON> @ Customer Support""", """Hello <PERSON>,
We're sorry to hear about the issues you're experiencing with your hearing aids. We have sent a return label to your email address <EMAIL_ADDRESS> and have initiated an exchange for both sets of hearing aids. Please check your inbox for instructions.
Best regards,
<PERSON> @ Customer Support
""")
    
# Score: tensor([1., 1., 0., 1., 1.])