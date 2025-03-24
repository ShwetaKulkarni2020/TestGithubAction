
import time
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TextEvals
from evidently.descriptors import Contains, TextLength, Sentiment, WordCount
from evidently.descriptors import *
#from TextSummary_CI.CD.GitExample.new1_working_code import *
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from torch.optim import AdamW
from torch.nn import functional as F
from evidently.metrics.custom_metric import CustomValueMetric
from evidently.renderers.html_widgets import WidgetSize
from evidently.ui.workspace.cloud import CloudWorkspace



ws = CloudWorkspace(token="dG9rbgET87IEpk9LnJbX/rvLPySLX/NTW2EcMIZiM8+VB+lY1gBQ9h/FTF8UImxCKyRDKxYGhXZCE2GFCvrEbsHRoqcCZZPNXVzHN04Nl7h8+7KlKDWkUwbwgTsreSe4IKtFSnTJCD453MurZLnKq4JNafptGGoMVfY9", 
                    url="https://app.evidently.cloud")

#project = ws.create_project("PROJECT1-GRP8")
#project.description = "Seamless Integration of Large Language Models in CI/CD Pipelines with Continuous Monitoring"
#project.save()

# Example usage
text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. NLP is a complex field that involves linguistics, computer science, and cognitive science. It includes tasks like machine translation, sentiment analysis, summarization, and question answering, which are essential for applications like voice assistants, chatbots, and recommendation systems.
"""
reference_summary = """
NLP is a field of AI that enables computers to understand, interpret, and generate human language. It includes tasks like machine translation, sentiment analysis, summarization, and question answering.
"""

# Initialize the model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def response_time(model_name):
    text = """
    Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. NLP is a complex field that involves linguistics, computer science, and cognitive science. It includes tasks like machine translation, sentiment analysis, summarization, and question answering, which are essential for applications like voice assistants, chatbots, and recommendation systems.
    """

    reference_summary = """
    NLP is a field of AI that enables computers to understand, interpret, and generate human language. It includes tasks like machine translation, sentiment analysis, summarization, and question answering.
    """
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    # Start the timer before generating the model response
    #start_time = time.time()
    generated_summary = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)
    #response_time = time.time() - start_time
    return summary
 
#Map your input data columns
column_mapping = ColumnMapping(
    
    text_features=['text'],
    
)



# Simulated data for evaluation
data = {
    "input_text": text,
    "ground_truth_summary": reference_summary
    
}

# Convert to DataFrame
df = pd.DataFrame(data,index=[0])

# Call response_time function to get summary and response time
summary = response_time(model_name)

# Add model-generated summaries to the dataframe
df["model_summary"] = summary

# Define column mapping for Evidently
column_mapping = ColumnMapping(target="ground_truth_summary", prediction="model_summary")

text_evals_report = Report(metrics=[
    
    TextEvals(column_name="model_summary",
              descriptors=[
                  TextLength(),
                  WordCount(),
                  Sentiment()
              ]
    ),
    
])


text_evals_report1 = Report(metrics=[
    TextEvals(column_name="model_summary",
              descriptors=[
                  IncludesWords(
                      words_list=['understand', 'interpret', 'generate'],
                      display_name="Mention definition of NLP"
                      )
                      ]
              )
])

#Semantic Similarity
text_evals_report2 = Report(metrics=[
    TextEvals(column_name="model_summary", descriptors=[
        SemanticSimilarity(with_column="input_text", 
                           display_name="Response-Question Similarity"),
    ])
])

text_evals_report2.run(reference_data=None,
                      current_data=df,
                      column_mapping=column_mapping)
text_evals_report2


text_evals_report.run(reference_data=None,
                      current_data=df,
                      column_mapping=column_mapping)

text_evals_report1.run(reference_data=None,
                      current_data=df,
                      column_mapping=column_mapping)

#print('Entered this section')
text_evals_report.save_html("file2_feb.html")
#text_evals_report1.save_html("file3_feb.html")
text_evals_report2.save_html("file4_feb.html")
#print('xited this section')

#projectid = '0193d8a6-43fc-7305-a029-56a3bfb33b19'
#ws.add_report(projectid, text_evals_report)
#ws.add_report(projectid, text_evals_report1)
#ws.add_report(projectid, text_evals_report2)