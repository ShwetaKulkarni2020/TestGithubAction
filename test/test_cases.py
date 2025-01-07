import pytest
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llm1 import train_rl, compute_rouge
#import sys
#sys.path.insert(0,'D:\MTech Projects\CapstoneProject\TextSummary_CI\CD\GitExample\src')
import sys
print(sys.path)



# Example usage
text1 = """
Sikkim, a small yet enchanting state in northeastern India, is a paradise for nature lovers and adventure enthusiasts. Surrounded by the mighty Himalayas, it offers stunning views of Kanchenjunga, the third-highest peak in the world. A visit to Gangtok, the capital, reveals a blend of modernity and tradition, with vibrant markets and serene monasteries. Tsomgo Lake, located at an altitude of 12,400 feet, is a must-see for its breathtaking beauty, while Yumthang Valley, the 'Valley of Flowers,' is renowned for its colorful flora. The state’s rich Tibetan Buddhist culture is evident in its monasteries, and visitors can also indulge in local delicacies like momos and phagshapa. Sikkim's serene villages like Pelling and Lachung provide perfect retreats, offering tranquility and opportunities for trekking."""

reference_summary1 = """
Sikkim is a scenic state in northeastern India known for its Himalayan views, including Kanchenjunga. Key attractions include Gangtok's mix of culture and modernity, Tsomgo Lake, Yumthang Valley, and monasteries showcasing Tibetan Buddhist heritage. The state also offers unique local cuisine and peaceful retreats in villages like Pelling and Lachung, making it ideal for nature lovers, trekkers, and cultural explorers."""


@pytest.mark.parametrize("num_epochs, lr", [(3, 1e-5), (5, 1e-4)])
def test_train_rl(num_epochs, lr):
    
    # ReInitialize the model and tokenizer
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Ensure that the model is in training mode (as the function expects training mode for backpropagation)
    model.train()

    # Run the RL training loop
    trained_model = train_rl(text1, reference_summary1, model, tokenizer, num_epochs=num_epochs, lr=lr)

    # Test if the model returns a trained model and not None
    assert trained_model is not None, "The trained model should not be None after training."

    # Test that the model's output is a non-empty summary
    inputs = tokenizer(text1, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    generated_summary = trained_model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)

    assert isinstance(summary, str), "The output summary should be a string."
    assert len(summary) > 0, "The generated summary should not be empty."
     # Test if the reward function is being applied correctly (ROUGE score)
    reward = compute_rouge(reference_summary1, summary)
    assert reward > 0, "The reward (ROUGE score) should be greater than zero, indicating meaningful summarization."

    print("\nTest completed successfully.")
