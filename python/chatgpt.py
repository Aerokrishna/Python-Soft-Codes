import openai
import re

# Initialize OpenAI API
openai.api_key = "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"

# Input natural language command
user_input = 'generate 10 samples'

# Use OpenAI GPT API to parse and convert to structured format
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "generate a dataset for training an nlp model. Data should consist of text describing checkpoints as alphabets for a robot to move. An array of checkpoints should be the output after processing the text"},
        {"role": "user", "content": user_input}
    ]
)

# Extract the structured response
structured_output = response['choices'][0]['message']['content']
print("Structured Output:\n", structured_output)
