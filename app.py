
from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# Initialize the Flask application
app = Flask(__name__)

# Define the model name
model_name = "google/pegasus-xsum"

# Load the tokenizer from the pre-trained Pegasus model
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Determine if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Pegasus model and move it to the specified device
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    """
    Handle the text summarization request.
    """
    if request.method == "POST":
        # Get the input text from the form
        inputtext = request.form["inputtext_"]
        
        # Prepare the text for summarization
        input_text = "summarize: " + inputtext
        
        # Tokenize the input text
        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        
        # Generate the summary
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        
        # Decode the generated summary
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    # Render the output page with the summary
    return render_template("output.html", data={"summary": summary})

if __name__ == '__main__':
    # Run the Flask application
    app.run()
