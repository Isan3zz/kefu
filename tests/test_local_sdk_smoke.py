"""Tiny random local weights test SDK offline APIs, NOT BGE model quality."""
import socket
import pytest

def test_cuda_oom_reports_explicit_cpu_restart_option():
    import torch
    from app.rag import model_memory_guard
    with pytest.raises(RuntimeError,match='MODEL_DEVICE=cpu'):
        with model_memory_guard(): raise torch.cuda.OutOfMemoryError('injected OOM without allocating GPU memory')

def test_sentence_transformers_local_loading(tmp_path,monkeypatch):
    from transformers import BertConfig,BertModel,BertForSequenceClassification,BertTokenizer
    from app.config import settings
    from app.rag import LocalModels
    embed=tmp_path/'embedding'; rerank=tmp_path/'reranker'
    vocab={t:i for i,t in enumerate(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','退','款','七','天'])}
    tokenizer=BertTokenizer(vocab=vocab)
    for path in (embed,rerank):
        path.mkdir(); tokenizer.save_pretrained(path)
    BertModel(BertConfig(vocab_size=len(vocab),hidden_size=1024,num_hidden_layers=1,num_attention_heads=8,intermediate_size=32,max_position_embeddings=32)).save_pretrained(embed)
    BertForSequenceClassification(BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=32,num_labels=1)).save_pretrained(rerank)
    def no_network(*args,**kwargs): raise AssertionError('Network access during local model loading')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    config=settings.model_copy(update={'embedding_model_path':str(embed),'reranker_model_path':str(rerank),'model_device':'cpu','embed_max_seq_length':16,'rerank_max_length':16})
    models=LocalModels(config)
    assert models.encode(['退款']).shape==(1,1024)
    assert len(models.score([('退款','七天退款')]))==1
