import discord
from transformers import pipeline
from dotenv import load_dotenv
import os

TOXICITY_THRESHOLD = 0.60
classifier = pipeline("text-classification", model="./goliath-model")

load_dotenv()
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

scores = {}

def predict_toxicity(text):
    result = classifier(text)[0]
    return result['score'] if result['label'] == 'toxic' else 1 - result['score']

async def moderate(message):
    score = predict_toxicity(message.content)
    scores[message.id] = score
    if score > TOXICITY_THRESHOLD:
        try:
            await message.delete()
            await message.channel.send(f"⚠️ Your message has been deleted because it was deemed too toxic ({score:.2f})")
        except discord.errors.Forbidden:
            print(f"[WARN] Pas la permission de supprimer le message de {message.author}")
    return score

@client.event
async def on_ready():
    print(f"Bot connecté : {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    score = await moderate(message)
    print(f"{message.author} -> {message.content} -> {score:.2f}")

client.run(os.getenv("TOKEN"))
