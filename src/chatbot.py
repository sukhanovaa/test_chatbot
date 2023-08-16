from typing import List, Tuple, Deque
import torch
import logging
from collections import deque
from time import time
from random import choice
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from metrics import ChatbotMetrics


logging.disable(logging.CRITICAL)


class Chatbot:
    __slots__ = ('_device', 'model', 'tok', 'max_len', 'bot_turns', 'user_turns', 'flow_cnt', 'history')
    
    # TODO: tune prefix
    # USR_PREFIX = 'USER: '
    # BOT_PREFIXES = ('YOU: ', 'YOU (flirting) ', 'YOU (lovingly): ')
    HOT_TOPIC: Tuple[str, ...] = (
         #"Barbie or Oppenheimer?",  -- need more fresh data that is probably not in any corpora yet
         "Have you ever taken a long hike?", 
         "What's your favourite drink?", 
         "Are you a Macaw, Parakeet, or Cacique?", 
         "Do you know what \"leucocholy\" means?"
         )

    def __init__(self, config):
        self._device = self._check_device()
        peft_config = PeftConfig.from_pretrained(config['model_weights'])
        self.model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                                          device_map=self._device)  # load_in_8bit=True
        self.model = PeftModel.from_pretrained(self.model, config['model_weights'])
        self.model.eval()

        self.tok = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        self.tok.pad_token = self.tok.eos_token
        self.max_len: int = AutoConfig.from_pretrained(peft_config.base_model_name_or_path).max_position_embeddings

        self.bot_turns: List[str] = []
        self.user_turns: List[str] = []
        self.flow_cnt: int = 0
        self.history: Deque[int] = deque(maxlen=self.max_len)

    @staticmethod
    def _check_device():
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    
    def _prepare_ids(self, raw_input: str):
        res = self.tok(raw_input, max_length=self.max_len, 
                       truncation=True, truncation_strategy='left', 
                       )['input_ids'] + [self.tok.eos_token_id]
        return res

    def _update_history_from_str(self, new_input: str):
        new_input = self._prepare_ids(new_input)
        self.history.extend(new_input)

    def _update_history_from_ids(self, input_ids: torch.Tensor):
        input_ids = input_ids.cpu().tolist()
        self.history.extend(input_ids) #  + [self.tok.eos_token_id])
    
    def _generate(self):
        res_ids = self.model.generate(input_ids=torch.tensor([self.history], device=self._device), 
                                      do_sample=True, 
                                      top_k=30, 
                                      top_p=0.95,
                                      temperature=0.8,
                                      pad_token_id=self.tok.eos_token_id,
                                      # min_new_tokens=10,
                                      max_new_tokens=50
                                      )[0][len(self.history):]
        self._update_history_from_ids(res_ids)
        res_text = self.tok.decode(res_ids, skip_special_tokens=True)
        self.bot_turns.append(res_text)
        return res_text
    
    def on_start(self):
        msg = f"Hello! I am a friendly bot, available for a quick chat. " \
              f"Message me! Or hit Ctrl+C when you get tired. By the way: \n"\
              f"{choice(self.HOT_TOPIC)}"
        self._update_history_from_str(msg)
        self.bot_turns.append(msg)
        return msg

    def respond(self, user_input: str):
        self.user_turns.append(user_input)
        self._update_history_from_str(user_input)
        self.flow_cnt += 1
        response = self._generate()
        # TODO on flow_cnt > 10, > 30
        return response

    def get_metrics(self):
        metrics = ChatbotMetrics(self.bot_turns, self.user_turns)
        metrics_msg = f"""Thanks for the chat!
        The chat lasted {metrics['num_replies']} lines. 
        Distinctness: {metrics['distinct']}. 
        Repeats after user: {metrics['repeats']}.
        """
        return metrics_msg
        
    def restart(self):
        # if necessary, these should be handled and saved before
        self.bot_turns = []
        self.user_turns = []
        self.flow_cnt = 0
        self.history = deque(maxlen=self.max_len)


# if __name__ == '__main__':
#     import yaml
#     with open('config.yaml') as inp:
#         config = yaml.load(inp, Loader=yaml.SafeLoader)
#     c = Chatbot(config)
#     print(c.on_start())
#     t = input()
#     print(c.respond(t))
#     print(c.bot_turns)
#     print(c.tok.decode(c.history))
