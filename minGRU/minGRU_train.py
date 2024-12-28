from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
loss_values = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
    
    avg_loss = epoch_loss / len(progress_bar)
    loss_values.append(avg_loss)

# Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
