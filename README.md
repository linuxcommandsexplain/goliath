# Goliath

Goliath is a Discord bot equipped with an AI with 125 million parameters based on the French model **textdetox/xlmr-large-toxicity-classifier**, which detects toxic messages (see video below). The model is 100% customizable. 

# Exemple

![Peek 18-02-2026 20-25](https://github.com/user-attachments/assets/e5c8e56f-5635-4149-9cdd-0c13bc2b46f5)


# Installation
Before you begin, make sure you have Python 3.14 installed on your computer. If not, you can [install it here](https://www.python.org/).

Create a `.env` file with the following value inside
```
TOKEN="YOUR_TOKEN"
```
*Replace `YOUR_TOKEN` with your bot's token, otherwise it won't work lol*

Then, in your terminal, enter the command: 
```bash
pip install requirements.txt
```
Then you can launch the bot and have fun with it.

# Training

For training, there is already a .csv training file for the French language, which you can easily replace with your own language.

Once you have done this, you can run this command: 
```bash
python tuning.py
```
What you are about to do is called **Fine Tuning**. The principle consists of taking an existing model and improving it with your own parameters. In my case, this involves adding parameters that define whether a particular word is toxic or not.

## /!\ rule for dataset.csv

The file format is as follows:
```
text    label
```
`Text` is your sentence.
`label` is the parameter that defines whether your sentence is an insult. 
`1` means it is toxic. 
`0` means it is not an insult.

As you can see, you can customize the IA model however you like.
