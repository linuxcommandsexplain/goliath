from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_NAME = "./goliath-model"

df = pd.read_csv("dataset.csv", sep="\t")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Texte": "text"})
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir="./goliath-model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=1,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Évaluation AVANT entraînement
print("\n" + "="*50)
print("📊 ÉVALUATION AVANT ENTRAÎNEMENT")
print("="*50)
metrics_before = trainer.evaluate()
for key, value in metrics_before.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

trainer.train()

# Évaluation APRÈS entraînement
print("\n" + "="*50)
print("📊 ÉVALUATION APRÈS ENTRAÎNEMENT")
print("="*50)
metrics_after = trainer.evaluate()
for key, value in metrics_after.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Comparaison
print("\n" + "="*50)
print("📈 COMPARAISON")
print("="*50)
for key in metrics_before:
    if isinstance(metrics_before[key], float):
        diff = metrics_after[key] - metrics_before[key]
        if "loss" in key or "runtime" in key:
            arrow = "✅ " if diff < 0 else "❌ +"
        else:
            arrow = "✅ +" if diff > 0 else "❌ "
        print(f"  {key}: {metrics_before[key]:.4f} → {metrics_after[key]:.4f} ({arrow}{diff:.4f})")
print("="*50 + "\n")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
rapport_path = f"./goliath-model/rapport_{timestamp}.txt"

with open(rapport_path, "w") as f:
    f.write(f"RAPPORT D'ENTRAÎNEMENT — {timestamp}\n")
    f.write(f"Modèle de base : {MODEL_NAME}\n")
    f.write(f"Dataset : {len(dataset['train']) + len(dataset['test'])} exemples "
            f"({len(dataset['train'])} train / {len(dataset['test'])} test)\n")
    f.write(f"Epochs : {args.num_train_epochs} | LR : {args.learning_rate} | Batch : {args.per_device_train_batch_size}\n")

    f.write("\n" + "="*50 + "\n")
    f.write("RÉSUMÉ EN CLAIR\n")
    f.write("="*50 + "\n")

    acc_before = metrics_before.get("eval_accuracy", 0)
    acc_after = metrics_after.get("eval_accuracy", 0)
    loss_before = metrics_before.get("eval_loss", 0)
    loss_after = metrics_after.get("eval_loss", 0)
    diff_acc = acc_after - acc_before

    f.write(f"\n🎯 Précision (accuracy)\n")
    f.write(f"   Avant : {acc_before*100:.1f}%  →  Après : {acc_after*100:.1f}%  (gain de {diff_acc*100:.1f} points)\n")
    f.write(f"   Avant l'entraînement, le modèle classifiait correctement {acc_before*100:.0f} textes sur 100.\n")
    f.write(f"   Après l'entraînement, il en classe correctement {acc_after*100:.0f} sur 100.\n")

    f.write(f"\n📉 Taux d'erreur (loss)\n")
    f.write(f"   Avant : {loss_before:.4f}  →  Après : {loss_after:.4f}  (réduction de {loss_before-loss_after:.4f})\n")
    f.write(f"   Plus cette valeur est basse, moins le modèle se trompe. Une bonne valeur est en dessous de 0.5.\n")

    if acc_after >= 0.90:
        appreciation = "Excellent entraînement, le modèle est très performant."
    elif acc_after >= 0.80:
        appreciation = "Bon entraînement. Le modèle est opérationnel mais peut encore progresser avec plus de données."
    elif acc_after >= 0.70:
        appreciation = "Entraînement correct mais perfectible. Essaie d'ajouter plus d'exemples au dataset."
    else:
        appreciation = "Entraînement insuffisant. Le dataset est probablement trop petit ou déséquilibré."

    f.write(f"\n💬 Conclusion\n")
    f.write(f"   {appreciation}\n")
    f.write(f"   Dataset utilisé : {len(dataset['train']) + len(dataset['test'])} exemples. ")
    if len(dataset['train']) + len(dataset['test']) < 2000:
        f.write("C'est un petit dataset, ajouter plus d'exemples améliorerait les résultats.\n")
    else:
        f.write("La taille du dataset est correcte.\n")

    f.write("\n" + "="*50 + "\n")
    f.write("CHIFFRES BRUTS\n")
    f.write("="*50 + "\n")
    f.write("\nAVANT\n")
    for key, value in metrics_before.items():
        f.write(f"  {key}: {value:.4f}\n" if isinstance(value, float) else f"  {key}: {value}\n")
    f.write("\nAPRÈS\n")
    for key, value in metrics_after.items():
        f.write(f"  {key}: {value:.4f}\n" if isinstance(value, float) else f"  {key}: {value}\n")

print(f"📄 Rapport sauvegardé : {rapport_path}")

model.save_pretrained("./goliath-model")
tokenizer.save_pretrained("./goliath-model")