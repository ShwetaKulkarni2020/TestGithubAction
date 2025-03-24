import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from torch.optim import AdamW
from torch.nn import functional as F
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TextEvals
from evidently.descriptors import TextLength

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

    # Encode input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
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
My Unforgettable Sikkim Trip Experience
Nestled in the lap of the Eastern Himalayas, Sikkim has always been a place of great fascination for me. The state, bordered by Bhutan, Tibet, and Nepal, is home to majestic mountains, serene lakes, vibrant monasteries, and a rich cultural heritage. My journey to this enchanting destination was one of discovery—of nature, adventure, and spirituality. The following is a vivid recount of my time in Sikkim, a place where every day brought new marvels, and each corner seemed to tell a different story.

Day 1: Arrival in Gangtok – A Land of Wonders
My Sikkim adventure began in Gangtok, the capital city, perched at an altitude of around 5,500 feet. As soon as I arrived, I was immediately charmed by its laid-back vibe and scenic beauty. The first thing that struck me was the breathtaking view of the Kangchenjunga range, the third highest mountain in the world, looming in the distance. I could already feel the presence of nature’s grandeur that would accompany me throughout my journey.

After checking into my hotel, I decided to explore the MG Marg, the heart of Gangtok. The cobbled streets, lively cafés, and bustling markets filled with local handicrafts were a delightful sight. I treated myself to a steaming plate of momos (dumplings) and thukpa (noodle soup) from one of the street vendors. These local delicacies were simple, yet so flavorful, and perfectly complemented the cool, crisp air.

In the evening, I took a peaceful walk to the Namgyal Institute of Tibetology, a must-visit place for those interested in learning about Tibetan culture and Buddhist art. The museum, filled with ancient manuscripts, thangkas, and sculptures, gave me a glimpse into the rich Tibetan Buddhist heritage that is deeply ingrained in Sikkim's identity. As night fell, the twinkling lights of Gangtok gave the town a magical charm.

Day 2: Exploring Tsomgo Lake and Baba Mandir
The following morning, I set off on a thrilling trip to Tsomgo Lake, located at an altitude of 12,313 feet. The scenic drive, which wound through lush forests and alpine meadows, was filled with twists and turns, making the journey as exciting as the destination itself. Tsomgo Lake, with its pristine waters surrounded by snow-capped mountains, looked like a place out of a fairy tale. The lake is considered sacred by the locals, and the surrounding area was decorated with colorful prayer flags that fluttered in the cold breeze.

Nearby, I visited Baba Mandir, dedicated to Baba Harbhajan Singh, a soldier of the Indian Army who is believed to have died in an accident, only to continue guarding the area in spirit. The temple was peaceful, and the devotion I witnessed there was palpable. I took a moment to reflect on the reverence the locals had for the soldier and the spirituality that permeates the entire region.

Day 3: Nathula Pass – Where India Meets China
The third day of my trip was undoubtedly one of the most thrilling and memorable moments of my journey. I took a ride to Nathula Pass, a high-altitude border pass that connects India to China. This pass, at 14,140 feet, is one of the highest motorable roads in the world. As we ascended towards Nathula, the roads became steeper, and the air thinner, but the surrounding beauty kept me captivated.

At the pass, I had the surreal experience of being so close to China, separated only by a few meters and a small border post. The landscape was rugged and harsh, yet it was beautiful in its starkness. The Indian Army soldiers stationed there were incredibly kind, and I had the privilege of interacting with them. Their stories of life at the border were humbling and gave me a deep appreciation for the sacrifices they make for the nation.

We also visited the Tibetan Border Police Memorial, a solemn place that pays tribute to those who lost their lives while serving at the border. The entire experience left me with a deep sense of respect for the men and women who safeguard the nation's frontiers.

Day 4: Spiritual Sojourn – Rumtek Monastery and Enchey Monastery
Sikkim is not just a haven for nature lovers, but it is also deeply spiritual. On the fourth day, I decided to explore the Rumtek Monastery, one of the largest and most famous monasteries in Sikkim. Located just outside Gangtok, it is a prominent center for Tibetan Buddhism. The monastery, with its colorful murals, towering prayer wheels, and chanting monks, radiated a sense of serenity that was hard to describe. The views of Gangtok from the monastery were equally stunning, offering a panoramic vista of the valley and the mountains.

After Rumtek, I visited the Enchey Monastery, perched on a hilltop near Gangtok. The peaceful ambiance and the stunning architecture of the monastery made it another unforgettable stop on my journey. I walked around the monastery, taking in the beauty of the surrounding landscape, while the sound of the prayer bells echoed through the air.

Day 5: Pelling – A Gateway to Kangchenjunga
After soaking in the beauty of Gangtok, I took a scenic drive to Pelling, located in the western part of Sikkim. Known for its panoramic views of Kangchenjunga, Pelling was a place of quiet reflection and natural beauty. The drive itself was an adventure, with winding roads that took me through thick forests and small villages. The final destination, however, was the view of Kangchenjunga—majestic, towering, and ever-present, it felt like the mountain was watching over the land.

I visited the Pemayangtse Monastery in Pelling, which is famous for its stunning architecture and the remarkable view it offers of the mountains. The monastery was peaceful, and I spent hours admiring the intricate sculptures and religious artifacts. I also visited Khecheopalri Lake, which is considered sacred by the local people. The lake, nestled among dense forests, is a place of solitude, where you can sit by the water and enjoy the serenity of the surroundings.

Day 6: Ravangla and Buddha Park – A Spiritual Farewell
On my final day in Sikkim, I traveled to Ravangla, a small town in southern Sikkim that offers some of the best views of Kangchenjunga. The town is home to the Buddha Park, a peaceful area where a giant statue of Buddha stands in the center, overlooking the valley. The park, with its well-maintained gardens and the serene atmosphere, was a perfect place to reflect on my journey.

The Buddha Statue at Ravangla is an iconic symbol of peace, and I spent a long time simply sitting and meditating, taking in the natural beauty of the surroundings. The view of Kangchenjunga, standing proudly in the distance, added to the majesty of the place. It was the perfect way to say goodbye to Sikkim—a place that had left an indelible mark on my heart.

Conclusion: A Journey to Remember
As I boarded my flight back home, I couldn’t help but feel a deep sense of gratitude. Sikkim had offered me so much—beautiful landscapes, rich cultural experiences, moments of adventure, and time for reflection. The people I met along the way, the stories I heard, and the landscapes I witnessed all combined to make my trip unforgettable.

Sikkim is a land where every corner has a story to tell. From the towering peaks of Kangchenjunga to the serene monasteries, from the bustling streets of Gangtok to the quiet, sacred lakes of the region, this place is truly one of the gems of India. I left with my heart full and my spirit refreshed, knowing that I had experienced something extraordinary. Sikkim had not only shown me its natural beauty, but also its heart—its rich culture, its warm hospitality, and its deep spirituality.

This trip will always remain one of the most memorable journeys of my life, and I hope to return to Sikkim again one day to explore more of its hidden treasures.
"""
reference_summary = """
My journey to Sikkim was a captivating blend of nature, adventure, and spirituality. Beginning in Gangtok, the capital, I explored the vibrant MG Marg, visited the Namgyal Institute of Tibetology, and marveled at the distant view of the Kangchenjunga range. The trip took me to Tsomgo Lake and Baba Mandir, sacred sites offering serene views and local reverence.

A visit to Nathula Pass, at the India-China border, was a highlight, where I interacted with soldiers and felt the grandeur of the place. Exploring Sikkim's spirituality, I visited the Rumtek and Enchey Monasteries, immersing myself in Tibetan Buddhist culture.

Pelling offered breathtaking views of Kangchenjunga, and I visited the Pemayangtse Monastery and Khecheopalri Lake for a peaceful experience. The final day took me to Ravangla and the Buddha Park, a place of reflection with a towering Buddha statue.

Sikkim's beauty, from snow-capped peaks to sacred lakes, coupled with its culture and warm hospitality, made it an unforgettable journey. The trip left me with a sense of deep appreciation and peace, marking it as one of the most memorable experiences of my life.

"""

# Fine-tune using RL
model = train_rl(text, reference_summary, model, tokenizer)

# Test the model after training
model.eval()
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
generated_summary = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)

print("\nFinal Summary after RL Fine-tuning:")
print(summary)

