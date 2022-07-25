Mappings = {
    "roberta": "transformers_bert_mapping",
    "bert": "transformers_bert_mapping",
    "vit": "transformers_vit_mapping",
    "albert": "transformers_albert_mapping",
    "gpt": "transformers_gpt_mapping",
    "clip": "transformers_clip_mapping",
    "bart": "transformers_bart_mapping",
    "t5": "transformers_t5_mapping",
}

transformers_albert_mapping = {
    "encoder": {"__name__":"encoder",
        "albert_layer_groups": {"__name__":"group",
            "$": {"__name__":"$",
                "albert_layers": {"__name__":"layer", 
                    "$": {"__name__":"$",
                        "attention": {"__name__":"self_attn",
                            "query": {"__name__":"q_proj"},
                            "key": {"__name__":"k_proj"},
                            "value": {"__name__":"v_proj"},
                            "dense": {"__name__":"out_proj"},
                            "LayerNorm": {"__name__":"layernorm"},
                        },
                        "ffn": {"__name__":"ffn.intermediate"},
                        "ffn_output": {"__name__":"ffn.output"},
                        "full_layer_layer_norm": {"__name__":"ffn.layernorm"},
                    },
                },
            }
        },
        "embedding_hidden_mapping_in": {"__name__": "hidden_mapping_in"},
    },
}

transformers_bert_mapping = {
    "encoder": {"__name__":"encoder",
        "layer": {"__name__":"layer",
            "$": {"__name__":"$",
                "attention": {"__name__":"self_attn",
                    "self.query": {"__name__":"q_proj"},
                    "self.key": {"__name__":"k_proj"},
                    "self.value": {"__name__":"v_proj"},
                    "output.dense": {"__name__":"out_proj"},
                    "output.LayerNorm": {"__name__":"layernorm"},
                },
                "intermediate.dense": {"__name__":"ffn.intermediate"},
                "output": {"__name__":"ffn",
                            "dense": {"__name__":"output"},
                            "LayerNorm": {"__name__":"layernorm"}
                },       
            }
        }
    },
}

transformers_vit_mapping = {
    "encoder": {"__name__":"encoder",
        "layer": {"__name__":"layer",
            "$": {"__name__":"$",
                "attention": {"__name__":"self_attn",
                    "attention.query": {"__name__":"q_proj"},
                    "attention.key": {"__name__":"k_proj"},
                    "attention.value": {"__name__":"v_proj"},
                    "output.dense": {"__name__":"out_proj"},
                },
                "layernorm_before": {"__name__":"self_attn.layernorm"},
                "intermediate.dense": {"__name__":"ffn.intermediate"},
                "output.dense": {"__name__":"ffn.output"},
                "layernorm_after": {"__name__":"ffn.layernorm"},       
            }
        }
    },
}

transformers_clip_mapping = {
    "logit_scale": {"__name__":"logit_scale"},
    "text_model": {"__name__":"text_model",
        "encoder": {"__name__":"encoder",
            "layers": {"__name__":"layer",
                "$": {"__name__":"$",
                    "self_attn": {"__name__":"self_attn",
                        "q_proj": {"__name__":"q_proj"},
                        "k_proj": {"__name__":"k_proj"},
                        "v_proj": {"__name__":"v_proj"},
                        "out_proj": {"__name__":"out_proj"},
                    },
                    "layer_norm1": {"__name__":"self_attn.layernorm"},
                    "mlp.fc1": {"__name__":"ffn.intermediate"},
                    "mlp.fc2": {"__name__":"ffn.output"},
                    "layer_norm2": {"__name__":"ffn.layernorm"},       
                }
            }
        },    
    },
    "vision_model": {"__name__":"vision_model",
        "encoder": {"__name__":"encoder",
            "layers": {"__name__":"layer",
                "$": {"__name__":"$",
                    "self_attn": {"__name__":"self_attn",
                        "q_proj": {"__name__":"q_proj"},
                        "k_proj": {"__name__":"k_proj"},
                        "v_proj": {"__name__":"v_proj"},
                        "out_proj": {"__name__":"out_proj"},
                    },
                    "layer_norm1": {"__name__":"self_attn.layernorm"},
                    "mlp.fc1": {"__name__":"ffn.intermediate"},
                    "mlp.fc2": {"__name__":"ffn.output"},
                    "layer_norm2": {"__name__":"ffn.layernorm"},       
                }
            }
        },    
    },
}

transformers_bart_mapping = {
    "encoder": {"__name__":"encoder",
        "layernorm_embedding": {"__name__":"embeddings.layernorm"},
        "embed_tokens": {"__name__":"embeddings.word_embeddings"},
        "embed_positions": {"__name__":"embeddings.position_embeddings"},
        "layers": {"__name__":"layer",
            "$": {"__name__":"$",
                "self_attn": {"__name__":"self_attn",
                    "q_proj": {"__name__":"q_proj"},
                    "k_proj": {"__name__":"k_proj"},
                    "v_proj": {"__name__":"v_proj"},
                    "out_proj": {"__name__":"out_proj"},
                },
                "self_attn_layer_norm": {"__name__":"self_attn.layernorm"},
                "fc1": {"__name__":"ffn.intermediate"},
                "fc2": {"__name__":"ffn.output"},
                "final_layer_norm": {"__name__":"ffn.layernorm"},       
            }
        }
    },

    "decoder": {"__name__":"decoder",
        "layernorm_embedding": {"__name__":"embeddings.layernorm"},
        "embed_tokens": {"__name__":"embeddings.word_embeddings"},
        "embed_positions": {"__name__":"embeddings.position_embeddings"},
        "layers": {"__name__":"layer",
            "$": {"__name__":"$",
                "self_attn": {"__name__":"self_attn",
                    "q_proj": {"__name__":"q_proj"},
                    "k_proj": {"__name__":"k_proj"},
                    "v_proj": {"__name__":"v_proj"},
                    "out_proj": {"__name__":"out_proj"},
                },
                "encoder_attn": {"__name__":"encoder_attn",
                    "q_proj": {"__name__":"q_proj"},
                    "k_proj": {"__name__":"k_proj"},
                    "v_proj": {"__name__":"v_proj"},
                    "out_proj": {"__name__":"out_proj"},
                },
                "self_attn_layer_norm": {"__name__":"self_attn.layernorm"},
                "encoder_attn_layer_norm": {"__name__":"encoder_attn.layernorm"},
                "fc1": {"__name__":"ffn.intermediate"},
                "fc2": {"__name__":"ffn.output"},
                "final_layer_norm": {"__name__":"ffn.layernorm"},            
            }
        }
    },    
}

transformers_t5_mapping = {
    "shared": {"__name__":"shared"},
    "encoder": {"__name__":"encoder",
        "block": {"__name__":"layer",
            "$": {"__name__":"$",
                "layer.0.SelfAttention": {"__name__":"self_attn",
                    "q": {"__name__":"q_proj"},
                    "k": {"__name__":"k_proj"},
                    "v": {"__name__":"v_proj"},
                    "o": {"__name__":"out_proj"},
                },
                "layer.0.layer_norm": {"__name__":"self_attn.layernorm"},
                "layer.1.DenseReluDense.wi": {"__name__":"ffn.intermediate"},
                "layer.1.DenseReluDense.wo": {"__name__":"ffn.output"},
                "layer.1.DenseReluDense.wi_0": {"__name__":"ffn.intermediate_0"},
                "layer.1.DenseReluDense.wi_1": {"__name__":"ffn.intermediate_1"},
                "layer.1.layer_norm": {"__name__":"ffn.layernorm"},
            }
        }
    },

    "decoder": {"__name__":"decoder",
        "block": {"__name__":"layer",
            "$": {"__name__":"$",
                "layer.0.SelfAttention": {"__name__":"self_attn",
                    "q": {"__name__":"q_proj"},
                    "k": {"__name__":"k_proj"},
                    "v": {"__name__":"v_proj"},
                    "o": {"__name__":"out_proj"},
                },
                "layer.1.EncDecAttention": {"__name__":"encoder_attn",
                    "q": {"__name__":"q_proj"},
                    "k": {"__name__":"k_proj"},
                    "v": {"__name__":"v_proj"},
                    "o": {"__name__":"out_proj"},
                },
                "layer.0.layer_norm": {"__name__":"self_attn.layernorm"},
                "layer.1.layer_norm": {"__name__":"encoder_attn.layernorm"},
                "layer.2.DenseReluDense.wi": {"__name__":"ffn.intermediate"},
                "layer.2.DenseReluDense.wo": {"__name__":"ffn.output"},
                "layer.2.DenseReluDense.wi_0": {"__name__":"ffn.intermediate_0"},
                "layer.2.DenseReluDense.wi_1": {"__name__":"ffn.intermediate_1"},                
                "layer.2.layer_norm": {"__name__":"ffn.layernorm"},
            }
        }
    },    
}

def convert_name(org_key, model_name, verbose=False):
    segments = org_key.split(".")
    query = ""
    if model_name not in Mappings:
        raise KeyError(model_name)
    node = eval(Mappings[model_name])

    new_segments = []
    for segment in segments:
        query += segment
        if query in node:
            node = node[query]
            new_segment = node["__name__"] if node["__name__"] != "" else query
            new_segments.append(new_segment)
            query = ""
        elif "$" in node:
            node = node["$"]
            new_segments.append(query)
            query = ""
        else:
            query += "." 
    if query!="":
        new_segments.append(query.strip(".")) # tailing query
    new_key = ".".join(new_segments)
    if verbose:
        print(f"{org_key} => {new_key}")
    return new_key
