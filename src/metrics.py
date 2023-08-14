from typing import List
from sacremoses import MosesTokenizer


class ChatbotMetrics:
    def __init__(self, sequences_bot: List[str], sequences_user: List[str], ngrams_order: int = 2):
        self.tok = MosesTokenizer('en')
        self.sequences_bot = [self.__preprocess(x) for x in sequences_bot]
        self.sequences_user = [self.__preprocess(x) for x in sequences_user]

        self.ngrams_order = ngrams_order

    def __preprocess(self, sequence: str):
        sequence = sequence.rsplit('|')[1]
        sequence = self.tok.tokenize(sequence.lower(), escape=False)
        return sequence

    def __ngrams(self, sequence: str):
        return [tuple(sequence[x: x+self.ngrams_order]) for x in range(len(sequence)-self.ngrams_order + 1)]

    def _calculate_distinctness(self):
        # % of unique ngrams (default n=2) in bot's responses
        all_responses = []
        for seq in self.sequences_bot:
            all_responses.extend(self.__ngrams(seq))
        unique = set(all_responses)
        if not all_responses:
            return None
        return len(unique) / len(all_responses)
        
    def _calculate_repeats(self):
        # average Jaccard over bot-user pairs
        pairs = list(zip(self.sequences_user, self.sequences_bot[1:]))
        paired_overlaps = []
        for (usr, bot) in pairs:
            usr = self.__ngrams(usr)
            bot = self.__ngrams(bot)
            overlap = len(set(usr) & set(bot))
            overall = len(usr + bot)
            paired_overlaps.append(overlap / overall)
        if not len(paired_overlaps):
            return None
        return sum(paired_overlaps) / len(paired_overlaps)
    
    def calculate_metrics(self):
        return {'repeats': self._calculate_repeats(), 
                'distinct': self._calculate_distinctness(), 
                "num_replies": len(self.sequences_user)}

# c = ChatbotMetrics(['| This is a start message', '| Hello! Not bad', '| thank you`'], ['| Hey! How are you?', "| this is great!"])
# print(c.calculate_metrics())