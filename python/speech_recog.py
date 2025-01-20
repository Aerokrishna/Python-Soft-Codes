import openai
import speech_recognition as sr

# OpenAI API key
openai.api_key = "your_openai_api_key"

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to recognize speech from the microphone
def recognize_speech():
    with sr.Microphone() as source:
        print("Please say your navigation command:")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Optional: Reduces background noise
        try:
            # Listen to the user's input
            audio = recognizer.listen(source)
            # Recognize speech using Google Web Speech API
            command = recognizer.recognize_google(audio)
            print(f"Recognized Command: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Error with the Speech Recognition service: {e}")
            return None

# Recognize speech and process it
user_input = recognize_speech()
if user_input:
    # Use OpenAI GPT API to process the recognized speech
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a robot navigation assistant that converts navigation commands to waypoints that need to be followed and the ones to be avoided in one word which is the name of the waypoint."},
            {"role": "user", "content": user_input}
        ]
    )

    # Extract and display the structured response
    structured_output = response['choices'][0]['message']['content']
    print("Structured Output:", structured_output)
