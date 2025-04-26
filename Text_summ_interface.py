import torch
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from torch.optim import AdamW
import warnings

# Suppress warnings
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
        labels = labels[:, :outputs.size(1)]  # Ensure labels are the same length as outputs
        outputs = model(input_ids=inputs['input_ids'], labels=labels)
        loss = outputs.loss

        # Scale the loss by the reward
        scaled_loss = loss * reward

        # Backpropagation and optimization step
        scaled_loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}/{num_epochs}, Summary: {summary}, Reward (ROUGE-1): {reward:.4f}, Loss: {scaled_loss.item():.4f}")

    return model

# Streamlit user interface
def main():
    st.title("Text Summarization using T5")

    # User input for text and reference summary
    input_text = st.text_area("Enter Text for Summarization", "")
    reference_summary = st.text_area("Enter Reference Summary (for ROUGE evaluation)", "")

    # Button to fine-tune the model using reinforcement learning
    if st.button("Fine-tune and Summarize"):
        if input_text and reference_summary:
            st.write("Fine-tuning model... This may take a few moments.")

            model_name = "t5-base"
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            # Fine-tune the model using reinforcement learning
            model = train_rl(input_text, reference_summary, model, tokenizer)

            # Test the model after training
            model.eval()
            structured_test_prompt = "Provide a structured summary of the following text with the following sections: Introduction, Main Points, Conclusion.\n\n"
            structured_test_text = structured_test_prompt + input_text

            inputs = tokenizer(structured_test_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            generated_summary = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)

            # Display the generated summary and ROUGE score
            st.write("Generated Summary:")
            st.write(summary)

            # Compute the ROUGE score
            rouge_score = compute_rouge(reference_summary, summary)
            st.write(f"ROUGE-1 Score: {rouge_score:.4f}")
        else:
            st.error("Please enter both text and reference summary.")

if __name__ == "__main__":
    main()