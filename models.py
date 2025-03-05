import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration

# Define configuration for the summarizer model (6 layers, 8 heads, 512 embed dim)
config = BartConfig(
    vocab_size=len(tokenizer),
    d_model=512,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    encoder_ffn_dim=2048,
    decoder_ffn_dim=2048,
    max_position_embeddings=1024,  # max input length
    eos_token_id=EOS_TOKEN_ID,
    bos_token_id=CLS_TOKEN_ID,
    pad_token_id=PAD_TOKEN_ID
)
summarizer = BartForConditionalGeneration(config)

# Weight initialization (Bart uses Xavier init by default for linear layers, embeddings are normed)
summarizer.tie_weights()  # ensure encoder/decoder embeddings are tied (as in standard Bart)

# Move model to GPU(s) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer.to(device)
if torch.cuda.device_count() > 1:
    summarizer = nn.DataParallel(summarizer)

# Define the summary encoder (6 layers, 8 heads, 512 dim) and document encoder (12 layers, 12 heads, 768 dim)
d_model_summary = 512
d_model_doc = 768
ffn_summary = 2048
ffn_doc = 3072

# Embeddings (token + positional) for summary and document encoders
summary_tok_embed = nn.Embedding(len(tokenizer), d_model_summary)
summary_pos_embed = nn.Embedding(1024, d_model_summary)
doc_tok_embed = nn.Embedding(len(tokenizer), d_model_doc)
doc_pos_embed = nn.Embedding(1024, d_model_doc)

# Initialize Transformer encoder layers
summary_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_summary, nhead=8, dim_feedforward=ffn_summary)
document_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_doc, nhead=12, dim_feedforward=ffn_doc)
summary_encoder_model = nn.TransformerEncoder(summary_encoder_layer, num_layers=6)
document_encoder_model = nn.TransformerEncoder(document_encoder_layer, num_layers=12)

# Utility: function to encode a sequence (returns [CLS] token's embedding)
def encode_sequence(input_ids, encoder_type="doc"):
    """
    Encode input_ids with the specified encoder.
    encoder_type: "doc" for document encoder, "summary" for summary encoder.
    Expects input_ids as a 2D tensor (batch, seq_len).
    Returns a tensor of shape (batch, hidden_dim) for [CLS] embeddings.
    """
    if encoder_type == "doc":
        tok_embed = doc_tok_embed
        pos_embed = doc_pos_embed
        encoder = document_encoder_model
    else:
        tok_embed = summary_tok_embed
        pos_embed = summary_pos_embed
        encoder = summary_encoder_model
    # Create positional ids (0,1,2,... up to seq_len-1)
    seq_len = input_ids.size(1)
    pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
    # Embed tokens and positions
    token_embeddings = tok_embed(input_ids)       # (batch, seq_len, d_model)
    position_embeddings = pos_embed(pos_ids)      # (batch, seq_len, d_model)
    embeddings = token_embeddings + position_embeddings
    # Permute to shape (seq_len, batch, d_model) for transformer
    embeddings = embeddings.transpose(0, 1)
    # Create attention mask (pad tokens)
    pad_mask = (input_ids == PAD_TOKEN_ID)        # boolean mask of shape (batch, seq_len)
    # Encode
    encoder_output = encoder(embeddings, src_key_padding_mask=pad_mask)  # (seq_len, batch, d_model)
    encoder_output = encoder_output.transpose(0, 1)  # back to (batch, seq_len, d_model)
    # Extract [CLS] embedding (we assume BOS token is at index 0 of each sequence)
    cls_embed = encoder_output[:, 0, :]  # (batch, d_model)
    return cls_embed

# Move encoders to device (and DataParallel if multi-GPU)
contrastive_params = list(summary_encoder_model.parameters()) + list(document_encoder_model.parameters()) + \
                     list(summary_tok_embed.parameters()) + list(summary_pos_embed.parameters()) + \
                     list(doc_tok_embed.parameters()) + list(doc_pos_embed.parameters())
contrastive_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_encoder_model.to(contrastive_device)
document_encoder_model.to(contrastive_device)
summary_tok_embed.to(contrastive_device)
summary_pos_embed.to(contrastive_device)
doc_tok_embed.to(contrastive_device)
doc_pos_embed.to(contrastive_device)
if torch.cuda.device_count() > 1:
    # We can wrap the encode_sequence function via DataParallel as needed, or handle splitting manually in training loop.
    summary_encoder_model = nn.DataParallel(summary_encoder_model)
    document_encoder_model = nn.DataParallel(document_encoder_model)
    summary_tok_embed = nn.DataParallel(summary_tok_embed)
    summary_pos_embed = nn.DataParallel(summary_pos_embed)
    doc_tok_embed = nn.DataParallel(doc_tok_embed)
    doc_pos_embed = nn.DataParallel(doc_pos_embed)

class WritingReviewer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512):
        super(WritingReviewer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # One-layer LSTM
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)  # to output a scalar score at each time step
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len) of token indices.
        Returns:
          scores: (batch, seq_len) of scores z_i for each prefix i.
          mean_score: (batch,) average of scores for each sequence.
        """
        # Embed tokens
        embeds = self.embed(input_ids)  # (batch, seq_len, embed_dim)
        # We can optionally incorporate a positional embedding or not (not explicitly stated, assume LSTM can handle sequence order).
        # Forward through LSTM
        output, _ = self.lstm(embeds)   # output shape: (batch, seq_len, hidden_dim)
        # Linear projection to scalar
        scores = self.linear(output).squeeze(-1)  # (batch, seq_len)
        # Compute mean score for each sequence
        # We should mask out padding positions to not include them in average.
        mask = (input_ids != PAD_TOKEN_ID).float()  # (batch, seq_len) mask of non-pad tokens
        lengths = mask.sum(dim=1)  # actual lengths of each sequence
        # avoid division by zero
        lengths[lengths == 0] = 1.0
        mean_score = (scores * mask).sum(dim=1) / lengths
        return scores, mean_score

# Instantiate the writing reviewer
writing_reviewer = WritingReviewer(vocab_size=len(tokenizer), embed_dim=512, hidden_dim=512)
writing_reviewer.to(device)
if torch.cuda.device_count() > 1:
    writing_reviewer = nn.DataParallel(writing_reviewer)
