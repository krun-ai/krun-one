import pytest
import torch

from krun_one.laya_adapter import LAYA_DIR
from krun_one.model import DecisionModel


needs_laya = pytest.mark.skipif(not (LAYA_DIR / "tokenizer").is_dir(),
                                reason="Laya checkpoint not found (set KRUN_PROJECT_ROOT to the dir holding laya/)")


@pytest.fixture(scope="session")
def tok():
    if not (LAYA_DIR / "tokenizer").is_dir():
        pytest.skip("Laya tokenizer not found (set KRUN_PROJECT_ROOT to the dir holding laya/)")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(LAYA_DIR / "tokenizer"))


def tiny_encoder_config(tmp_path):
    from transformers import ModernBertConfig

    cfg = ModernBertConfig(vocab_size=50368, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                           num_attention_heads=2, max_position_embeddings=1024, global_attn_every_n_layers=3,
                           local_attention=32, pad_token_id=50283, cls_token_id=50281, sep_token_id=50282,
                           bos_token_id=50281, eos_token_id=50282)
    cfg.save_pretrained(tmp_path)
    return tmp_path / "config.json"


@pytest.fixture
def tiny_model(tmp_path):
    from krun_one.model import build_encoder

    torch.manual_seed(0)
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    tiny_encoder_config(enc_dir)
    return DecisionModel(build_encoder(enc_dir), head_layers=1, n_act=2), enc_dir
