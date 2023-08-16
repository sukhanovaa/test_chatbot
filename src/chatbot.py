from torch import device
from torch.cuda import is_available
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
from metrics import ChatbotMetrics
from time import time
from random import choice
from itertools import chain


class Chatbot:
    __slots__ = ('_device', 'model', 'tok', 'max_len', 'bot_turns', 'user_turns', 'flow_cnt', 'bot_prefix')
    
    # TODO: no prefixes + avoid repetition
    USR_PREFIX = 'USER says: '
    BOT_PREFIXES = ('YOU reply: ', 'YOU reply (flirting) ', 'YOU reply (lovingly): ')
    HOT_TOPIC = ( #"Barbie or Oppenheimer?",  -- need more fresh data that is probably not in any corpora yet
                 "Have you ever taken a long hike?", 
                 "What's your favourite drink?", 
                 "Are you a Macaw, Parakeet, or Cacique?", 
                 "By the way, do you know what \"leucocholy\" means?")

    def __init__(self, config):
        self._device = self._check_device()
        peft_config = PeftConfig.from_pretrained(config['model_weights'])
        self.model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                                          device_map=self._device)  # load_in_8bit=True
        self.model = PeftModel.from_pretrained(self.model, config['model_weights'])
        self.model.eval()

        self.tok = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        self.max_len = AutoConfig.from_pretrained(peft_config.base_model_name_or_path).max_position_embeddings

        self.bot_turns = []
        self.user_turns = []
        self.flow_cnt = 0
        # self.user_times = []

        self.bot_prefix = 0

    @staticmethod
    def _check_device():
        if is_available():
            return device('cuda:0')
        else:
            return device('cpu')
        
    def __get_prefix_user(self, input_sequence: str):
        return self.USR_PREFIX + input_sequence

    def __get_prefix_bot(self, input_sequence = ""):
        return self.BOT_PREFIXES[self.bot_prefix] + input_sequence

    def on_start(self):
        msg = f"""Hello! I am a friendly bot, available for a quick chat. Message me! Or hit Ctrl+C when you get tired.
        {choice(self.HOT_TOPIC)}
"""
        self.bot_turns.append(self.__get_prefix_bot() + msg)
        return msg
    
    def _gather_history(self):
        return ' | '.join(chain.from_iterable(zip(self.bot_turns, self.user_turns))) + ' | ' + self.__get_prefix_bot()


    def _generate(self, history):
        prefix = self.tok(history, max_length=self.max_len, truncation=True, 
                          truncation_strategy='left', return_tensors='pt')['input_ids']
        res_ids = self.model.generate(input_ids=prefix.to(self._device), max_new_tokens=50)  # min_new_tokens=10,
        res_text = self.tok.decode(res_ids[0][prefix.shape[1]:], skip_special_tokens=True)
        return res_text

    def respond(self, user_input: str):
        self.user_turns.append(self.__get_prefix_user(user_input))
        self.flow_cnt += 1
        response = self._generate(self._gather_history())  # temperature, repetition_penalty, ...
        self.bot_turns.append(self.__get_prefix_bot() + response)
        # on flow_cnt > 10, > 30
        if self.flow_cnt == 10:
            self.bot_prefix += 1
        elif self.flow_cnt == 30:
            self.bot_prefix += 1
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


if __name__ == '__main__':
    import yaml
    with open('config.yaml') as inp:
        config = yaml.load(inp, Loader=yaml.SafeLoader)
    c = Chatbot(config)
    print(c.on_start())
    t = input()
    print(c.respond(t))
    print(c.bot_turns)