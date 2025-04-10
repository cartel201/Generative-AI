from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token to eos_token (works well for GPT-2)
tokenizer.pad_token = tokenizer.eos_token

# Load the Wikitext-2 dataset (using the "train" split)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Limit the dataset to the first N examples (change this value as needed)
dataset_limit = 200
dataset = dataset.select(range(dataset_limit))

# Preprocess the dataset: combine all texts into one large string.
def preprocess_dataset(dataset):
    filtered_texts = [item["text"] for item in dataset if item["text"].strip()]
    combined_text = " ".join(filtered_texts)
    return combined_text

train_text = preprocess_dataset(dataset)

# Tokenize the dataset with padding and truncation (adjust max_length as needed)
inputs = tokenizer(train_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the GPU (or CPU if no GPU)

# Define a custom PyTorch Dataset class
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Move each batch item to the specified device (GPU or CPU)
        return {key: val[idx].to(device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create a DataLoader with a small batch size for quick training
batch_size = 4
text_dataset = TextDataset(inputs)
dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

# Fine-tuning setup
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Learning rate scheduler
num_training_steps = len(dataloader) * 1  # 1 epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Mixed precision training (for faster training on GPU)
scaler = torch.cuda.amp.GradScaler()

# Training loop for 1 epoch
for epoch in range(5):
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()

        # Ensure the batch is on the correct device (should already be handled in our Dataset)
        batch = {key: val.to(device) for key, val in batch.items()}

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=batch['input_ids'], 
                labels=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            loss = outputs.loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

# Text Generation Section
model.eval()

# Ensure user input is handled properly
prompt = input('Enter Anything: ')  # Get user prompt for generation

# Tokenize the prompt with padding/truncation and move to GPU
input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)  # Create attention mask

# Generate text with the explicit attention mask
generated_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=1000,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# Decode and print the generated text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGenerated Text:\n", generated_text)
print(torch.cuda.is_available())  # Should return True if a GPU is available.
print(torch.cuda.current_device())  # Should return the current GPU device ID.