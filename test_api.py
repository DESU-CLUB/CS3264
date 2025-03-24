import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.environ["GROQ_API_KEY"])

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Explaining the concept of a neural network"}
    ],
    model="llama3-70b-8192",
    stream=True,
    temperature=0.5,
)

for chunk in chat_completion:
    print(chunk.choices[0].delta.content, end="")
