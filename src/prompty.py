import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from torch.optim import AdamW
import warnings

warnings.filterwarnings("ignore", message="You are using the default legacy behaviour")
# Initialize the model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Reward function based on ROUGE score
def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rouge1"].fmeasure  # You can change this to another ROUGE metric

# Define the RL training loop
def train_rl(text, reference_summary, model, tokenizer, num_epochs=3, lr=1e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    # Add structured prompt to the input text
    structured_prompt = "Provide a structured summary of the following text with the following sections: Introduction, Main Points, Conclusion.\n\n"
    structured_text = structured_prompt + text
    # Encode input text
    inputs = tokenizer(structured_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(reference_summary, return_tensors="pt", max_length=150, truncation=True, padding="max_length").input_ids
    
    # Move tensors to the device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    labels = labels.to(device)

    # Create the RL loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
          # Generate a summary (policy output)
        outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

        # Decode the output
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute the reward (ROUGE score)
        reward = compute_rouge(reference_summary, summary)

        # Calculate the loss
        # We can use the cross-entropy loss for training, and apply the reward during backpropagation
        # Here, we treat the reward as a scaling factor for the cross-entropy loss
        labels = labels[:, :outputs.size(1)]  # Ensure labels are the same length as outputs

        # Compute the loss using the standard cross-entropy loss between the generated and reference summary
        # We need to calculate loss using the labels and outputs (not reward directly)
        outputs = model(input_ids=inputs['input_ids'], labels=labels)
        loss = outputs.loss  # This is the cross-entropy loss from the model

        # Scale the loss by the reward
        # We multiply the loss by the reward to reinforce the learning (this is a typical RL approach)
        scaled_loss = loss * reward

        # Backpropagation and optimization step
        scaled_loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}/{num_epochs}, Summary: {summary}, Reward (ROUGE-1): {reward:.4f}, Loss: {scaled_loss.item():.4f}")

    return model
# Example usage
text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. NLP is a complex field that involves linguistics, computer science, and cognitive science. It includes tasks like machine translation, sentiment analysis, summarization, and question answering, which are essential for applications like voice assistants, chatbots, and recommendation systems.
"""
reference_summary = """
NLP is a field of AI that enables computers to understand, interpret, and generate human language. It includes tasks like machine translation, sentiment analysis, summarization, and question answering.
"""

# Fine-tune using RL
model = train_rl(text, reference_summary, model, tokenizer)

# Test the model after training
model.eval()
# Add structured prompt to the test input
test_text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. NLP is a complex field that involves linguistics, computer science, and cognitive science. It includes tasks like machine translation, sentiment analysis, summarization, and question answering, which are essential for applications like voice assistants, chatbots, and recommendation systems.
"""
structured_test_prompt = "Provide a structured summary of the following text with the following sections: Introduction, Main Points, Conclusion.\n\n"
structured_test_text = structured_test_prompt + test_text

inputs = tokenizer(structured_test_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
generated_summary = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)

print("\nFinal Summary after RL Fine-tuning:")
print(summary)