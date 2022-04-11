#
# Created by djz on 2022/04/01.
#
import enum
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_CAUSAL_LM_MAPPING
from .base import Pipeline

class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class TextGenerationPipeline(Pipeline):
    XL_PREFIX = """
    In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The
    voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western
    Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision
    and denounces one of the men as a horse thief. Although his father initially slaps him for making such an
    accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop,
    begging for his blessing. <eod> </s> <eos>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "prefix" not in self._preprocess_params:
            # This is very specific. The logic is quite complex and needs to be done
            # as a "default".
            # It also defines both some preprocess_kwargs and generate_kwargs
            # which is why we cannot put them in their respective methods.
            prefix = None
            if self.model.config.prefix is not None:
                prefix = self.model.config.prefix
            if prefix is None and self.model.__class__.__name__ in [
                "XLNetLMHeadModel",
                "TransfoXLLMHeadModel",
                "TFXLNetLMHeadModel",
                "TFTransfoXLLMHeadModel",
            ]:
                # For XLNet and TransformerXL we add an article to the prompt to give more state to the model.
                prefix = self.XL_PREFIX
            if prefix is not None:
                # Recalculate some generate_kwargs linked to prefix.
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        **generate_kwargs
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=False, return_tensors=self.framework
            )
            prefix_length = prefix_inputs["input_ids"].shape[-1]

            if "max_new_tokens" in generate_kwargs:
                pass
            elif "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += prefix_length
            else:
                generate_kwargs["max_length"] = self.model.config.max_length + prefix_length

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length
        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params

    # overriding _parse_and_tokenize to allow for unusual language-modeling tokenizer arguments
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kwargs.update({"add_space_before_punct_symbol": True})

        return super()._parse_and_tokenize(*args, **kwargs)

    def __call__(self, text_inputs, **kwargs):
        return super().__call__(text_inputs, **kwargs)

    def preprocess(self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs):
        inputs = self.tokenizer(
            prefix + prompt_text, padding=False, add_special_tokens=False, return_tensors= 'pt'
        )
        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.model.config.max_length) - cur_len
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")
        generated_sequence = self.model.generate(input_ids=input_ids, **generate_kwargs)  # BS x SL
        out_b = generated_sequence.shape[0]
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])

        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": generated_sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                if return_type == ReturnType.FULL_TEXT:
                    all_text = prompt_text + text[prompt_length:]
                else:
                    all_text = text[prompt_length:]

                record = {"generated_text": all_text}
            records.append(record)

        return records
