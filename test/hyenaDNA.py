from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, logging
import torch
import torch.utils.data.dataset as Dataset

device = torch.device("cuda", 3)

# instantiate pretrained model
checkpoint = '/mnt/sdb/home/lrl/code/git_hyenaDNA/hyenadna-tiny-1k-seqlen-d256-hf'
max_length = 160_000

# bfloat16 for better speed and reduced memory usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,  trust_remote_code=True)

# Generate some random sequence and labels
# If you're copying this code, replace the sequences and labels
# here with your own data!
sequence = 'ACTG' * int(max_length/4)
sequence = [sequence] * 8  # Create 8 identical samples
tokenized = tokenizer(sequence,truncation=True,return_tensors="pt")
labels = [0, 1] * 4

input_ids = tokenized['input_ids']
representation = model(input_ids=input_ids)
print(representation)


# # Create a dataset for training
# ds = Dataset.from_dict({"input_ids": tokenized, "labels": labels})
# ds.set_format("pt")
#
# # Initialize Trainer
# # Note that we're using extremely small batch sizes to maximize
# # our ability to fit long sequences in memory!
# args = {
#     "output_dir": "tmp",
#     "num_train_epochs": 1,
#     "per_device_train_batch_size": 1,
#     "gradient_accumulation_steps": 4,
#     "gradient_checkpointing": True,
#     "learning_rate": 2e-5,
# }
# training_args = TrainingArguments(**args)
#
# trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# result = trainer.train()
#
# print(result)
#
# # Now we can save_pretrained() or push_to_hub() to share the trained model!
