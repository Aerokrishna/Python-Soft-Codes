import openai
import re

# Initialize OpenAI API
openai.api_key = "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"

# Input natural language command
user_input = input("Enter your navigation command: ")

# Use OpenAI GPT API to parse and convert to structured format
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a robot navigation assistant that converts navigation commands to waypoints in the form of x,y"},
        {"role": "user", "content": user_input}
    ]
)

# Extract the structured response
structured_output = response['choices'][0]['message']['content']
print("Structured Output:\n", structured_output)

# Function to extract points from the structured output
def parse_waypoints(output):
    points = []
    # Regex to find all patterns like (x, y)
    matches = re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", output)
    for match in matches:
        x, y = map(float, match)  # Convert strings to floats
        points.append((x, y))
    return points

# Convert the structured output into an array of goal points
goal_points = parse_waypoints(structured_output)
print("Goal Points:", goal_points)
