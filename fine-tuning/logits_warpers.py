import torch
from transformers import LogitsWarper, LogitsProcessor
from typing import Callable, Iterable, List, Optional


# modified from HF https://github.com/huggingface/transformers/blob/5041bc3511d098814598cf1cfc6c6bd20e72c144/src/transformers/generation_logits_process.py#L287
def _get_ngrams_as_list(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    generated_ngrams = []
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        cur_ngrams = list(zip(*[gen_tokens[i:] for i in range(ngram_size)]))
        generated_ngrams.append(cur_ngrams)
    return generated_ngrams


class NoPresentKeyphraseLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of keyphrases in the one2seq setting.

    Args:
        max_encoder_ngram_size (`int`): max length of keyphrase n-gram to check
        encoder_input_ids (`int`):
            The encoder_input_ids that contain phrases not to be repeated within the decoder ids.
        sep_token (`int`): separator token
        prefix_len (`int`): number of leading special tokens in the decoder's output
    """
    def __init__(self, max_encoder_ngram_size, encoder_input_ids,
                 sep_token, eos_token, prefix_len, tokenizer=None):
        if max_encoder_ngram_size <= 0:
            raise ValueError(
                f"`max_encoder_ngram_size` has to be a strictly positive integer, but is {max_encoder_ngram_size}"
            )

        # This breaks HF's design but is necessary if we want to block phrase w/ & w/o whitespace
        self.tokenizer = tokenizer
        
        self.sep_token = sep_token
        self.eos_token = eos_token
        self.prefix_len = prefix_len

        self.max_ngram_size = max_encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]

        # get all ngrams to block
        self.input_ngrams = [[] for _ in range(self.batch_size)]
        for n in range(self.max_ngram_size):
            cur_ngrams = _get_ngrams_as_list(n+1, encoder_input_ids, self.batch_size)
            for i in range(self.batch_size):
                self.input_ngrams[i] += cur_ngrams[i]

        self.input_ngrams = [set(x) for x in self.input_ngrams]

        # expand to block variants with and without preceding blank space
        '''
        if self.tokenizer is not None:
            for i in range(self.batch_size):
                expanded_phrases = [x[1:] for x in self.tokenizer.batch_decode(self.input_ngrams[i]) if x[0] == ' ']
                self.input_ngrams[i].update(set([tuple(x) for x in tokenizer(expanded_phrases, add_special_tokens=False)['input_ids']]))      
        '''
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]

        if cur_len > self.prefix_len:
            examples_to_ban_sep_token = []
            for i in range(num_batch_hypotheses):
                cur_generated_ids = input_ids[i][self.prefix_len:].tolist()
                cur_generated_ids_str = ' '.join((str(x) for x in cur_generated_ids))
                cur_generated_phrases = [x.strip() for x in
                                         cur_generated_ids_str.split(str(self.sep_token))]
                prev_phrases = set(cur_generated_phrases[:-1])
                cur_partial_phrase = cur_generated_phrases[-1]
                cur_partial_phrase = tuple([int(x.strip()) for x in cur_partial_phrase.split()])
                
                if cur_partial_phrase in self.input_ngrams[i]:
                    examples_to_ban_sep_token.append(i)
                # special handling for the first phrase
                elif len(prev_phrases) == 0:                    
                    if tuple(self.tokenizer(' ' + self.tokenizer.decode(cur_partial_phrase), add_special_tokens=False)['input_ids']) in self.input_ngrams[i]:
                        examples_to_ban_sep_token.append(i)

            # block the separator token and the eos token
            scores[examples_to_ban_sep_token, self.sep_token] = -float("inf")
            scores[examples_to_ban_sep_token, self.eos_token] = -float("inf")

            # renormalize
            scores = scores.log_softmax(dim=-1)
        
        return scores
       
       
class NoRepeatKeyphraseLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces that no generated keyphrase repeats previous keyphrases.

    Args:
        sep_token (`int`): separator token
        prefix_len (`int`): number of leading special tokens in the decoder's output
    """

    def __init__(self, sep_token, prefix_len, eos_token=None):
        self.sep_token = sep_token
        self.prefix_len = prefix_len
        self.eos_token = eos_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        
        if cur_len - self.prefix_len > 1:
            examples_to_ban_sep_token = []
            for i in range(num_batch_hypotheses):
                cur_generated_ids = input_ids[i][self.prefix_len:].tolist()
                cur_generated_ids_str = ' '.join((str(x) for x in cur_generated_ids))
                cur_generated_phrases = [x.strip() for x in
                                         cur_generated_ids_str.split(str(self.sep_token))]
                if len(cur_generated_phrases) == 1:
                    continue
                else:
                    prev_phrases = set(cur_generated_phrases[:-1])
                    cur_partial_phrase = cur_generated_phrases[-1]
                    if cur_partial_phrase in prev_phrases:
                        examples_to_ban_sep_token.append(i)
                        # print(input_ids[i].tolist(), prev_phrases, cur_partial_phrase)
                        # print("=================")


            scores[examples_to_ban_sep_token, self.sep_token] = -float("inf")
            if self.eos_token is not None:
                scores[examples_to_ban_sep_token, self.eos_token] = -float("inf")

        # renormalize
        scores = scores.log_softmax(dim=-1)
            
        return scores


# from a newer version of huggingface 
class TypicalLogitsWarper(LogitsWarper):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
