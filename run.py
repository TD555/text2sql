from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("Tigran555/text2sql")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.do(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Example query
query = "What is the average salary of employees in department 50?"

# Tokenize query
input_ids = tokenizer.encode(query, return_tensors="pt").to(device)

# Generate SQL query․ The max_length parameter is 20 by default, it should be changed for the number of generated tokens
output_ids = model.generate(input_ids)

# Decode generated SQL query
sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated SQL Query:", sql_query)