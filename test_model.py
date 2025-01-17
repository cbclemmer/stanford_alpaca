import argparse
import time
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

input_file = 'test_input.txt'
instruction_file = 'test_instruction.txt'
model_folder = 'fine_tuned_model'

if not os.path.exists(input_file):
    print(f"The file test_input.txt does not exist.")
    exit()

if not os.path.exists(instruction_file):
    print(f"The file test_instruction.txt does not exist.")
    exit()

if not os.path.exists(model_folder):
    print(f"The model folder {model_folder} does not exist")
    exit()

print("Loading model from", model_folder)
# Load the pre-trained model and move it to the specified device
model_load_t = time.time()
model = AutoModelForCausalLM.from_pretrained('fine_tuned_model').to(device)
print(f"Model took {time.time() - model_load_t:.2f}s to load")

print("Loading tokenizer from", model_folder)
tokenizer_load_t = time.time()
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model')
print(f'Tokenizer took {time.time() - tokenizer_load_t:.2f}s to load')

# Define the pipeline to generate text
print('Loading generator')
generator_load_t = time.time()
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
print(f'Generator took {time.time() - generator_load_t:.2f}s to load')

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
[INSTRUCTION]

### Input:
[INPUT]

### Response:
"""

# Load the prompt from a file
with open(input_file, "r") as f:
    prompt_input = f.read()

print("Loading test instruction from", 'test_instruction.txt')
# Load the test instruction from a file and replace [INPUT] and [INSTRUCTION] with the prompt and instruction text, respectively
with open(instruction_file, "r") as f:
    instruction = f.read()

prompt_template = prompt_template.replace("[INSTRUCTION]", instruction)
prompt_template = prompt_template.replace("[INPUT]", prompt_input)

print("Generating text...")
# Generate text based on the formatted test instruction
text_generation_load_t = time.time()
output = generator(prompt_template, max_length=500, do_sample=True, temperature=0.7)
print(f'Text generation took {time.time() - text_generation_load_t:.2f}s to load')

# Print the generated text to stdout
print("Generated text:")
print(output[0]['generated_text'])

print(f'Total time: {time.time() - model_load_t:.2f}s')
