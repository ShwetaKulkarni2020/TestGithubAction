import time
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.custom_metric import CustomValueMetric
from evidently.renderers.html_widgets import WidgetSize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from evidently.descriptors import Contains, TextLength, Sentiment, WordCount
from evidently.ui.workspace.cloud import CloudWorkspace
from transformers import pipeline
import gradio as gr
#from transformers import BartForConditionalGeneration

ws = CloudWorkspace(token="dG9rbgET87IEpk9LnJbX/rvLPySLX/NTW2EcMIZiM8+VB+lY1gBQ9h/FTF8UImxCKyRDKxYGhXZCE2GFCvrEbsHRoqcCZZPNXVzHN04Nl7h8+7KlKDWkUwbwgTsreSe4IKtFSnTJCD453MurZLnKq4JNafptGGoMVfY9", 
                    url="https://app.evidently.cloud")

# Example usage
text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. NLP is a complex field that involves linguistics, computer science, and cognitive science. It includes tasks like machine translation, sentiment analysis, summarization, and question answering, which are essential for applications like voice assistants, chatbots, and recommendation systems.
"""
reference_summary = """
NLP is a field of AI that enables computers to understand, interpret, and generate human language. It includes tasks like machine translation, sentiment analysis, summarization, and question answering.
"""

# Initialize the model and tokenizer for DistilBART
# Step 1: Load DistilBART model and tokenizer for summarization
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#model = BartForConditionalGeneration.from_pretrained(model_name)
        
    
    


def token_count(summary):
    return len(summary.split()) 

#def model_size(model):
 #   num_parameters = sum(p.numel() for p in model.parameters())
  #  return num_parameters

def response_time(model_name):
    text = """
    Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. NLP is a complex field that involves linguistics, computer science, and cognitive science. It includes tasks like machine translation, sentiment analysis, summarization, and question answering, which are essential for applications like voice assistants, chatbots, and recommendation systems.
    """

    reference_summary = """
    NLP is a field of AI that enables computers to understand, interpret, and generate human language. It includes tasks like machine translation, sentiment analysis, summarization, and question answering.
    """
    # Start the timer before generating the model response
    start_time = time.time()
     # Generate summary using the DistilBART summarizer
    generated_summary = summarizer(text, max_length=200, min_length=80, do_sample=False)
    summary = generated_summary[0]["summary_text"]
    response_time = time.time() - start_time
    # Print the summary
    print(summary)
    return response_time, summary

def model_score(reference_summary, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, summary)
    return scores['rouge1'].fmeasure

# Simulated data for evaluation
data = {
    "input_text": text,
    "ground_truth_summary": reference_summary
}

# Convert to DataFrame
df = pd.DataFrame(data, index=[0])

# Call response_time function to get summary and response time
response_time_value, summary = response_time(model_name)

# Add model-generated summaries to the dataframe
df["model_summary"] = summary

# Define column mapping for Evidently
column_mapping = ColumnMapping(target="ground_truth_summary", prediction="model_summary")

# Define the Report with Custom Value Metric
text_evals_report = Report(metrics=[
    
    CustomValueMetric(func=lambda data: token_count(summary), title="TOKEN COUNT", size=WidgetSize.HALF),
   # CustomValueMetric(func=model_size, title="MODEL SIZE", size=WidgetSize.HALF),
    CustomValueMetric(func=lambda data: response_time_value, title="RESPONSE TIME", size=WidgetSize.HALF),
    CustomValueMetric(func=lambda data: model_score(reference_summary, summary), title="ROUGE SCORE", size=WidgetSize.HALF)
])

# Run the report
text_evals_report.run(reference_data=None, current_data=df, column_mapping=column_mapping)

# Save the report to an HTML file
text_evals_report.save_html("Model3_file.html")

#projectid = '0193882b-bfff-7ee5-9d00-692a86e16cd8'

#ws.add_report(projectid, text_evals_report)