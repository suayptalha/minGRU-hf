def train_model(model, tokenizer, train_data, output_dir, epochs=3, batch_size=16, learning_rate=5e-5, block_size=128):
    train_dataset = TinyStoriesDataset(train_data, tokenizer, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            batch = batch.to(device)

            outputs = model(batch, labels=batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

    model.save_pretrained(output_dir, safe_serialization = False)
    tokenizer.save_pretrained(output_dir)
