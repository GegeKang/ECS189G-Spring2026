import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

local_dir = './Qwen3-0.6B-Base'
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.float32, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

# your_review_here = 'Cannot find any redeeming qualities in this film.'
# your_review_here = 'Could not be better.'
# your_review_here = 'The movie was a complete disaster from start to finish.'
# your_review_here = "Not what I expected, but I suppose it was fine."
your_review_here = "I will give it 6/10. Good actors but boring plot. "
in_context_prompt = f"""Review: I loved this movie, it was fantastic!
Sentiment: Positive

Review: Terrible acting and boring plot.
Sentiment: Negative

Review: A visual masterpiece with a gripping storyline.
Sentiment: Positive

Review: Completely unwatchable from start to finish.
Sentiment: Negative

Review: {your_review_here}
Sentiment:"""

input_ids = tokenizer(in_context_prompt, return_tensors="pt").to(model.device)

max_new_tokens = 10
output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
print(tokenizer.decode(output[0], skip_special_tokens=False))