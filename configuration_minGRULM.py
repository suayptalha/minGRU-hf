from transformers import PretrainedConfig

class MinGRULMConfig(PretrainedConfig):
    model_type = "mingru"

    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        ff_mult=4,
        min_gru_expansion=1.5,
        expand=2.0,
        depth=12,
        n_layer=12,
        pad_vocab_size_multiple=8,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ff_mult = ff_mult
        self.min_gru_expansion = min_gru_expansion
        self.expand = expand
        self.depth = depth
        self.n_layer = n_layer
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.initializer_range = initializer_range
        super().__init__(**kwargs)
