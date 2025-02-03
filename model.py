
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torch.amp import GradScaler, autocast
# from datasets import load_dataset
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# from torch.nn.utils.rnn import pad_sequence
# import time
# from tqdm.auto import tqdm
# tqdm.pandas()

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Tokenizer
# class SimpleTokenizer:
#     def __init__(self):
#         self.special_tokens = {"[PAD]": 0, "[START]": 1, "[END]": 2}
#         self.tokens = {}
#         self.inverse_tokens = {}

#     def build_vocab(self, data):
#         vocab = set()
#         for example in data:
#             prompt, code = example
#             vocab.update(prompt.split())
#             vocab.update(code.split())
#         vocab = sorted(vocab)
#         for i, token in enumerate(vocab, start=3):
#             self.tokens[token] = i
#         self.inverse_tokens = {v: k for k, v in {**self.special_tokens, **self.tokens}.items()}

#     def encode(self, text, max_length=128):
#         tokens = [self.special_tokens["[START]"]]
#         for word in text.split():
#             tokens.append(self.tokens.get(word, len(self.tokens) + 1))
#         tokens = tokens[:max_length - 1]
#         tokens.append(self.special_tokens["[END]"])
#         tokens += [self.special_tokens["[PAD]"]] * (max_length - len(tokens))
#         return tokens

#     def decode(self, token_ids):
#         result = []
#         for id in token_ids:
#             if id == self.special_tokens["[END]"]:
#                 break
#             if id != self.special_tokens["[PAD]"]:
#                 result.append(self.inverse_tokens.get(id, "[UNK]"))
#         return " ".join(result)

#     def __len__(self):
#         return len(self.special_tokens) + len(self.tokens)

# # Dataset
# class CodeGenDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=128):
#         self.data = [{"input_ids": tokenizer.encode(prompt, max_length), "labels": tokenizer.encode(code, max_length)} for prompt, code in data]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # Collate function for DataLoader
# def collate_fn(batch):
#     input_ids = pad_sequence([torch.tensor(item["input_ids"]) for item in batch], batch_first=True, padding_value=0)
#     labels = pad_sequence([torch.tensor(item["labels"]) for item in batch], batch_first=True, padding_value=0)
#     return {"input_ids": input_ids, "labels": labels}

# # Model
# class CodeGenModel(nn.Module):
#     def __init__(self, vocab_size, d_model, n_heads, n_layers):
#         super(CodeGenModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True), num_layers=n_layers
#         )
#         self.decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True), num_layers=n_layers
#         )
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt):
#         src_emb = self.embedding(src)
#         tgt_emb = self.embedding(tgt)
#         enc_out = self.encoder(src_emb)
#         dec_out = self.decoder(tgt_emb, enc_out)
#         return self.fc_out(dec_out)

# # Training
# def train_model(model, train_loader, optimizer, criterion, scheduler, scaler, epochs):
#     model.train()
#     for epoch in range(epochs):
#         epoch_loss = 0
#         for _, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch: {epoch}")):
#             input_ids = batch["input_ids"].to(device)
#             labels = batch["labels"].to(device)

#             optimizer.zero_grad()
#             with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
#                 outputs = model(input_ids, input_ids)
#                 loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()

#             epoch_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader)}")

# # Evaluation
# def evaluate_model(model, tokenizer, test_data, max_length=128):
#     model.eval()
#     bleu_scores = []
#     rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
#     rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

#     with torch.no_grad():
#         for nl, code in test_data:
#             input_ids = torch.tensor([tokenizer.encode(nl, max_length)]).to(device)
#             tgt = torch.zeros((1, max_length), dtype=torch.long).to(device)
#             outputs = model(input_ids, tgt)
#             predictions = torch.argmax(outputs, dim=-1)
#             generated_code = tokenizer.decode(predictions[0].tolist())

#             bleu_scores.append(sentence_bleu([code.split()], generated_code.split(), smoothing_function=SmoothingFunction().method1))
#             scores = rouge_scorer_obj.score(code, generated_code)
#             for key in rouge_scores:
#                 rouge_scores[key] += scores[key].fmeasure

#     avg_bleu = sum(bleu_scores) / len(bleu_scores)
#     avg_rouge = {key: value / len(test_data) for key, value in rouge_scores.items()}
#     print(f"BLEU: {avg_bleu:.4f}, ROUGE: {avg_rouge}")
#     return avg_bleu, avg_rouge

# # Main
# if __name__ == "__main__":
#     # Hyperparameters
#     d_model = 512
#     n_heads = 8
#     n_layers = 6
#     max_len = 128
#     batch_size = 32
#     epochs = 10
#     learning_rate = 5e-4 # 3e-4 maximum, 1e-5 minimum
#     dropout_rate = 0.1
#     gradient_clipping = 1.0

#     # Load and preprocess dataset
#     start_time = time.time()
#     data_files = {"train": "train.json", "validation": "valid.json", "test": "test.json"}
#     dataset = load_dataset("json", data_files=data_files)
#     tokenizer = SimpleTokenizer()
#     tokenizer.build_vocab([(item["nl"], item["code"]) for item in dataset["train"]])

#     train_dataset = CodeGenDataset([(item["nl"], item["code"]) for item in dataset["train"]], tokenizer, max_length=max_len)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)

#     # Initialize model and optimizer
#     model = CodeGenModel(vocab_size=len(tokenizer), d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
#     criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["[PAD]"])
#     scaler = GradScaler(enabled=torch.cuda.is_available())

#     # Train model
#     train_model(model, train_loader, optimizer, criterion, scheduler, scaler, epochs)

#     # Evaluate model
#     test_data = [(item["nl"], item["code"]) for item in dataset["test"]]
#     evaluate_model(model, tokenizer, test_data, max_length=max_len)
#     print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")

#2nd version start
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from torch.nn.utils.rnn import pad_sequence
import time
from tqdm.auto import tqdm
import random
import pandas as pd  # For tabular score representation
import torch.optim as optim  # Ensure transformers is installed

# -------------------------------
# 1. Set Random Seeds for Reproducibility
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------
# 2. Set Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 3. Tokenizer with Few-Shot Capability
# -------------------------------
class SimpleTokenizer:
    def __init__(self):
        # Initialize special tokens
        self.special_tokens = {
            "[PAD]": 0,
            "[START]": 1,
            "[END]": 2,
            "[UNK]": 3,
            "[EXAMPLE]": 4  # New token for indicating examples
        }
        self.tokens = {}
        self.inverse_tokens = {}
    
    def build_vocab(self, data):
        """
        Build vocabulary from the dataset.

        Args:
            data (list of tuples): List of (prompt, code) tuples.
        """
        vocab = set()
        for example in data:
            prompt, code = example
            vocab.update(prompt.split())
            vocab.update(code.split())
        vocab = sorted(vocab)
        for i, token in enumerate(vocab, start=5):  # Start from 5 to account for special tokens
            self.tokens[token] = i
        self.inverse_tokens = {v: k for k, v in {**self.special_tokens, **self.tokens}.items()}
    
    def encode(self, text, max_length=512):
        """
        Encode a single prompt or code snippet.

        Args:
            text (str): The text to encode.
            max_length (int): Maximum sequence length.

        Returns:
            list: List of token IDs.
        """
        tokens = [self.special_tokens["[START]"]]
        for word in text.split():
            tokens.append(self.tokens.get(word, self.special_tokens["[UNK]"]))
        tokens = tokens[:max_length - 2]  # Reserve space for [END]
        tokens.append(self.special_tokens["[END]"])
        tokens += [self.special_tokens["[PAD]"]] * (max_length - len(tokens))
        return tokens
    
    def encode_few_shot(self, examples, current_prompt, max_length=512):
        """
        Encode multiple examples along with the current prompt for few-shot learning.

        Args:
            examples (list of tuples): List of (prompt, code) tuples.
            current_prompt (str): The current prompt for which to generate code.
            max_length (int): Maximum sequence length.

        Returns:
            list: List of token IDs.
        """
        tokens = [self.special_tokens["[START]"]]
        for prompt, code in examples:
            tokens.append(self.special_tokens["[EXAMPLE]"])
            for word in prompt.split():
                tokens.append(self.tokens.get(word, self.special_tokens["[UNK]"]))
            tokens.append(self.special_tokens["[END]"])
            for word in code.split():
                tokens.append(self.tokens.get(word, self.special_tokens["[UNK]"]))
            tokens.append(self.special_tokens["[END]"])
        # Add the current prompt
        tokens.append(self.special_tokens["[START]"])
        for word in current_prompt.split():
            tokens.append(self.tokens.get(word, self.special_tokens["[UNK]"]))
        tokens = tokens[:max_length - 2]  # Reserve space for [END]
        tokens.append(self.special_tokens["[END]"])
        tokens += [self.special_tokens["[PAD]"]] * (max_length - len(tokens))
        return tokens
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs back to text.

        Args:
            token_ids (list): List of token IDs.

        Returns:
            str: Decoded text.
        """
        result = []
        for id in token_ids:
            if id == self.special_tokens["[END]"]:
                break
            if id != self.special_tokens["[PAD]"] and id != self.special_tokens["[EXAMPLE]"]:
                result.append(self.inverse_tokens.get(id, "[UNK]"))
        return " ".join(result)
    
    def __len__(self):
        return len(self.special_tokens) + len(self.tokens)

# -------------------------------
# 4. Dataset with Few-Shot Examples
# -------------------------------
class CodeGenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, few_shot_k=5):
        """
        Initialize the dataset with few-shot examples.

        Args:
            data (list of tuples): List of (prompt, code) tuples.
            tokenizer (SimpleTokenizer): The tokenizer instance.
            max_length (int): Maximum sequence length.
            few_shot_k (int): Number of few-shot examples to include.
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.few_shot_k = few_shot_k
        
        for idx, (prompt, code) in enumerate(data):
            # Select few_shot_k examples randomly from the dataset
            if idx < few_shot_k:
                few_shot_examples = data[:idx] + data[idx+1:][:few_shot_k - idx]
            else:
                few_shot_examples = random.sample(data[:idx], self.few_shot_k)
            
            src = tokenizer.encode_few_shot(few_shot_examples, prompt, max_length=self.max_length)
            tgt = tokenizer.encode(code, max_length=self.max_length)
            # For decoder input, prepend [START] token and remove the last token
            decoder_input = [tokenizer.special_tokens["[START]"]] + tgt[:-1]
            self.data.append({
                "src": src,
                "tgt": decoder_input,
                "labels": tgt
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------------
# 5. Collate Function for DataLoader
# -------------------------------
def collate_fn(batch):
    """
    Collate function to pad sequences in the batch.

    Args:
        batch (list): List of data points.

    Returns:
        dict: Dictionary containing padded src, tgt, and labels.
    """
    src_ids = pad_sequence([torch.tensor(item["src"]) for item in batch], batch_first=True, padding_value=0)
    tgt_ids = pad_sequence([torch.tensor(item["tgt"]) for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item["labels"]) for item in batch], batch_first=True, padding_value=0)
    return {"src": src_ids, "tgt": tgt_ids, "labels": labels}

# -------------------------------
# 6. Transformer-Based Code Generation Model
# -------------------------------
class CodeGenModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout=0.1):
        """
        Initialize the Transformer-based Code Generation Model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Embedding dimension.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of Transformer layers.
            dropout (float): Dropout rate.
        """
        super(CodeGenModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2048, dropout=dropout, activation='relu', batch_first=True),
            num_layers=n_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=2048, dropout=dropout, activation='relu', batch_first=True),
            num_layers=n_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt):
        """
        Forward pass of the model.

        Args:
            src (Tensor): Source sequences [batch_size, src_seq_length].
            tgt (Tensor): Target sequences [batch_size, tgt_seq_length].

        Returns:
            Tensor: Output logits [batch_size, tgt_seq_length, vocab_size].
        """
        src_emb = self.embedding(src)  # [batch_size, src_seq_length, d_model]
        tgt_emb = self.embedding(tgt)  # [batch_size, tgt_seq_length, d_model]
        
        enc_out = self.encoder(src_emb)  # [batch_size, src_seq_length, d_model]
        
        # Generate subsequent mask for decoder to prevent attending to future tokens
        tgt_seq_length = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_length).to(tgt.device)
        
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)  # [batch_size, tgt_seq_length, d_model]
        dec_out = self.dropout(dec_out)
        output = self.fc_out(dec_out)  # [batch_size, tgt_seq_length, vocab_size]
        return output

# -------------------------------
# 7. Early Stopping Mechanism
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0.0, path='best_model.pt'):
        """
        Initialize Early Stopping.

        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        """
        Call method to check if early stopping should be triggered.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model to save if validation loss decreases.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model to save.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

# -------------------------------
# 8. Training Function
# -------------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, epochs, device, gradient_clipping, early_stopping):
    """
    Train the model with mixed precision and early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        scheduler (Scheduler): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision.
        epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        gradient_clipping (float): Maximum norm for gradient clipping.
        early_stopping (EarlyStopping): Early stopping mechanism.
    """
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for _, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch: {epoch+1}/{epochs}")):
            src = batch["src"].to(device)  
            tgt = batch["tgt"].to(device)  
            labels = batch["labels"].to(device)  

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(src, tgt)  
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")

        # ✅ Initialize avg_val_loss before validation
        val_loss = 0
        avg_val_loss = float('inf')

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(src, tgt)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        # ✅ Now that avg_val_loss is computed, we can step the scheduler
        scheduler.step(avg_val_loss)

        # ✅ Early Stopping Check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break


# -------------------------------
# 9. Evaluation Function
# -------------------------------
def evaluate_model(model, tokenizer, test_data, max_length=512, device='cpu', few_shot_k=5):
    """
    Evaluate the model on the test set and compute BLEU and ROUGE scores.

    Args:
        model (nn.Module): The trained model.
        tokenizer (SimpleTokenizer): The tokenizer instance.
        test_data (list of tuples): List of (prompt, code) tuples.
        max_length (int): Maximum sequence length.
        device (torch.device): Device to perform evaluation on.
        few_shot_k (int): Number of few-shot examples to include.

    Returns:
        tuple: Average BLEU and ROUGE scores.
    """
    model.eval()
    bleu_scores = []
    rouge_scores_total = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothing_fn = SmoothingFunction().method1

    with torch.no_grad():
        for idx, (nl, code) in enumerate(tqdm(test_data, desc="Evaluating")):
            # Select few_shot_k examples from the test data excluding the current one
            few_shot_examples = []
            if few_shot_k > 0 and len(test_data) > 1:
                # Simple strategy: use the first few examples as few-shot
                if len(test_data) > few_shot_k:
                    few_shot_examples = test_data[:few_shot_k]
                else:
                    few_shot_examples = test_data[:len(test_data)-1]
            
            src = torch.tensor([tokenizer.encode_few_shot(few_shot_examples, nl, max_length=max_length)]).to(device)  # [1, src_seq_length]
            tgt = torch.tensor([tokenizer.encode("[START]", max_length=max_length)]).to(device)  # [1, 1]

            generated = []
            for _ in range(max_length - 1):
                outputs = model(src, tgt)  # [1, tgt_seq_length, vocab_size]
                next_token_logits = outputs[0, -1, :]  # [vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1]
                if next_token.item() == tokenizer.special_tokens["[END]"]:
                    break
                generated.append(next_token.item())
                tgt = torch.cat([tgt, next_token], dim=1)  # [1, tgt_seq_length + 1]

            generated_code = tokenizer.decode(generated)

            # Compute BLEU
            bleu_score = sentence_bleu([code.split()], generated_code.split(), smoothing_function=smoothing_fn)
            bleu_scores.append(bleu_score)

            # Compute ROUGE
            scores = rouge_scorer_obj.score(code, generated_code)
            for key in rouge_scores_total:
                rouge_scores_total[key] += scores[key].fmeasure

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {key: (value / len(test_data)) if len(test_data) > 0 else 0 for key, value in rouge_scores_total.items()}
    
    print("\n=== Evaluation Metrics ===")
    
    # Create a DataFrame for the scores
    metrics_data = {
        "Metric": ["BLEU-1", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "Score (%)": [
            round(avg_bleu * 100, 2),  # Convert to percentage
            round(avg_rouge["rouge1"] * 100, 2),
            round(avg_rouge["rouge2"] * 100, 2),
            round(avg_rouge["rougeL"] * 100, 2)
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))

    return avg_bleu, avg_rouge

# -------------------------------
# 10. Main Execution
# -------------------------------
if __name__ == "__main__":
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    d_model = 768
    n_heads = 8
    n_layers = 6
    max_len = 256  # Reduced from 512 for debugging; adjust as needed
    batch_size = 8
    epochs = 5
    learning_rate = 2e-5  # Reduced from 1e-4
    dropout_rate = 0.3
    gradient_clipping = 2.0
    patience = 3  # Early Stopping patience
    few_shot_k = 5  # Number of few-shot examples

    # -------------------------------
    # Load and Preprocess Dataset
    # -------------------------------
    start_time = time.time()
    data_files = {"train": "train.json", "validation": "valid.json", "test": "test.json"}
    dataset = load_dataset("json", data_files=data_files)
    #print(dataset['train'])
    #exit("exit the program")
    # -------------------------------
    # Initialize and Build Tokenizer
    # -------------------------------
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([(item["nl"], item["code"]) for item in dataset["train"]])
    print(f"Vocabulary Size: {len(tokenizer)}")
    
    # -------------------------------
    # Create Datasets with Few-Shot Examples
    # -------------------------------
    train_dataset = CodeGenDataset(
        [(item["nl"], item["code"]) for item in dataset["train"]],
        tokenizer,
        max_length=max_len,
        few_shot_k=few_shot_k
    )
    val_dataset = CodeGenDataset(
        [(item["nl"], item["code"]) for item in dataset["validation"]],
        tokenizer,
        max_length=max_len,
        few_shot_k=few_shot_k
    )
    test_dataset = CodeGenDataset(
        [(item["nl"], item["code"]) for item in dataset["test"]],
        tokenizer,
        max_length=max_len,
        few_shot_k=few_shot_k
    )

    # -------------------------------
    # Create DataLoaders
    # -------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # -------------------------------
    # Initialize Model
    # -------------------------------
    model = CodeGenModel(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout_rate
    ).to(device)

    # -------------------------------
    # Initialize Optimizer
    # -------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )


    # -------------------------------
    # Initialize Scheduler (OneCycleLR)
    # -------------------------------
    total_steps = len(train_loader) * epochs
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=learning_rate,
    #     total_steps=total_steps
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=0.5, 
        patience=2
    )


    # -------------------------------
    # Initialize Loss Function with Label Smoothing and ignore_index
    # -------------------------------
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=tokenizer.special_tokens["[PAD]"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=tokenizer.special_tokens["[PAD]"])

    # -------------------------------
    # Initialize GradScaler for Mixed Precision
    # -------------------------------
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # -------------------------------
    # Initialize Early Stopping
    # -------------------------------
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path='best_model.pt'
    )

    # -------------------------------
    # Train the Model with Early Stopping
    # -------------------------------
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        scaler,
        epochs,
        device,
        gradient_clipping,
        early_stopping
    )

    # -------------------------------
    # Load the Best Model
    # -------------------------------
    model.load_state_dict(torch.load('best_model.pt',weights_only=True))

    # -------------------------------
    # Evaluate the Model
    # -------------------------------
    test_data = [(item["nl"], item["code"]) for item in dataset["test"]]
    avg_bleu, avg_rouge = evaluate_model(
        model,
        tokenizer,
        test_data,
        max_length=max_len,
        device=device,
        few_shot_k=few_shot_k
    )

    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")

#2nd version end