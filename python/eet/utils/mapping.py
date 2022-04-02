Mappings = {
    "RobertaModel": "transformers_bert_mapping",
    "BertModel": "transformers_bert_mapping",
    "ViTModel": "transformers_vit_mapping",
    "AlbertModel": "transformers_albert_mapping",
    "GPTModel": "transformers_gpt_mapping",
    "CLIPModel": "transformers_clip_mapping",
}

transformers_bert_mapping = {
    # "embeddings.word_embeddings": {"__name__":""},
    # "embeddings.position_embeddings": {"__name__":""},
    # "embeddings.token_type_embeddings": {"__name__":""},
    # "embeddings.LayerNorm": {"__name__":""},
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
