from transformers import PretrainedConfig

class MinGRUConfig(PretrainedConfig):
    model_type = "mingru"

    def __init__(
        self,
        vocab_size=30522,
        d_model=512, 
        ff_mult=4,
        expansion_factor=2,
        depth=12,          
        n_layer=12,        
        pad_vocab_size_multiple=8,  
        initializer_range=0.02,      
        hidden_size=512,   
        hidden_dropout_prob=0.1,  
        num_labels=2,      
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ff_mult = ff_mult
        self.expansion_factor = expansion_factor
        self.depth = depth
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_labels = num_labels
        super().__init__(**kwargs)
