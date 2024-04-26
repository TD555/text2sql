from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("Tigran555/text2sql")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.do(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Example db_name, schema and query

db_name = "product_catalog"

schema = """| attribute_definitions : attribute_id , attribute_name , attribute_data_type | catalogs : catalog_id , catalog_name , catalog_publisher , date_of_publication , date_of_latest_revision | catalog_structure : catalog_level_number , 
            catalog_id , catalog_level_name | catalog_contents : catalog_entry_id , catalog_level_number , parent_entry_id , previous_entry_id , next_entry_id , catalog_entry_name , product_stock_number , price_in_dollars , price_in_euros , price_in_pounds , capacity , 
            length , height , width | catalog_contents_additional_attributes : catalog_entry_id , catalog_level_number , attribute_id , attribute_value | catalog_structure.catalog_id = catalogs.catalog_id | 
            catalog_contents.catalog_level_number = catalog_structure.catalog_level_number | catalog_contents_additional_attributes.catalog_level_number = catalog_structure.catalog_level_number | catalog_contents_additional_attributes.catalog_entry_id = catalog_contents.catalog_entry_id |"""

query = 'Which catalog contents have a product stock number that starts from "2"? Show the catalog entry names.'

# Concatenate database name and schema with the query
input_text = " ".join([db_name, schema, question])

# Tokenize query
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate SQL query․ The max_length parameter is 20 by default, it should be changed for the number of generated tokens
output_ids = model.generate(input_ids)

# Decode generated SQL query
sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated SQL Query:", sql_query)
