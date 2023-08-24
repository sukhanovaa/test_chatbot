WIP

I was given a short amount of time to build an open-conversation chatbot. I'd like to note that, while I knew of big LLMs and adaptive finetuning, I could not get my hands on these concepts (also because of inadequate hardware). I also never had to build everything in 5 hours, and it shows. 

```
docker compose up
<configure training in model_finetune.ipynb>
python3 model_interact.py
```


I started with


## Exploring the metrics
What I considered:
1. Naive, regarding interaction statistics
    * Length of conversation: how long (in user lines) lasts a chat on average
    * Time between replies to the chatbot, which could indicate the level of a user's engagement. Requires analytics to establish thresholds.
    * Human feedback, as in
        - "Were you satisfied by the conversation? Y/N" at the end of the conversation - does not give fine-grained information about what happened during the conversation that led to a particular assessment
        - grade 1-10 - just as above. 
        - This is overall messy, but it makes sense for such feedback to be collected, then post-analyzed by assessors.
    * How often do users come back to chat again? Need users!
    * How often is a chat with a model abandoned at the start? Need users!


2. Naive, based on text
    * repetition / fluency, as in N of distinct n-grams - the only quantifiable metric. However, it really is just proxy metrics for generation quality; does not measure the adequacy of the conversation...
    * perplexity on the user questions? - this is not about answers' quality at all; besides, I doubt that huge models have difficulties in this domain
    * overlap between a user's question and a model's answer - I think this is a bit outdated for LLM evaluation, since it used to be a problem around 2017-2020 maybe. Could make sense if we have enough time to check the LM's answer before outputting it to the user.
    * F-measure/BLEU/etc. on questions with pre-defined answers? - might be good for tracking factual information, but the good chit-chat has nothing pre-defined a priori. There are separate tasks that could be used to see the prompted model performance though, e.g. summarization.
    * Embedding distance between the user's question and the model's answer - seems pretty much the same as the previous idea
    * % of negative user responses ("no, that's not what I was talking about"; "that's a dumb response"; "bad bot"); % of positive user responses ("You're funny!", "Great, thanks", "you're right")
        - very hard to define 'negative', but theoretically it could be (embedding) cosine similarity to sets of responses similar to above; results in a problem of constructing such sets, extracting user responses from paragraphs, etc.
        - metrics such as these are really very much user-dependent (what if a user is often ironic? what if they're displaying opinions far from a bot's training data distribution? what if a user is in a bad mood...)
        - but __could be captured__ if reframed as an entailment task (LM-user answer pairs, entailment/contradiction/neutral; we're looking for contradictions; something like https://arxiv.org/abs/1904.03371 ?) or as a sentiment analysis task. This requires additional models, though.

I actually expect that interaction-based metrics per model are much more telling than the textual ones. In the absence of definition for a good interaction, what should one compute? The definition lies in a field of discourse (and not in purely linguistic studies), after all.


There are also specialized metrics involving various aspects of quality, e.g. empathy: https://github.com/Sea94/ieval, which I think deserve attention.


Then I decided to

## Choose a model

A recent causal LM that fits my memory is all I ask, so under 13B
* OPT? Comes in flavours like < 1B, 1.3B, 2.7B, 6.7B
* GPT-J? 6B
* LLaMA? 7B

I looked at https://weightwatcher.ai/leaderboard.html and https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard and compared the top models regarding their Alpha and Average, respectively. Frankly, these two metrics seem to be contradicting each other. 

Surprisingly, an OPT-1.3B model was pretty small, high in Alpha and decent in terms of Truthfulness, although other metrics were not as impressive. Since I haven't had any opinion based on experience, I decided to test the pipeline on a small OPT. 

If I had had enough time/compute, I'd switch to the fresh LLaMA-2 (7B) -- it demonstrates a good average on the leaderboard tasks. As it turned out, Colab had a hard time supporting even 1.3 billion parameters.

I chose LoRA simply because this is the most simple concept of 'adapting' to me. Prefix tuning seems to be much more promising, though, since it might serve as a proxy task definition in the absence of seq2seq-like approach.

After that, it was time to

## Choose datasets & prepare them

Personally, I think that all dialogue corpora ought to be curated. As a source of data, Reddit probably contains most lively responses, but it is potentially unpredictable and should be tested first. 

I considered the following datasets:

* ConvAI2/3 dataset - on the closer look, it has a deeper structure that is useful in RLHF or evaluation contexts
* The NPS Chat Corpus - not in Huggingface Datasets, but potentially useful
* Cornell Movie-Dialogs Corpus
* HC3 (which also contains ChatGPT answers, but we won't need them)
* and DailyDialogue corpus


And then the problems started.

- I haven't opened a PR yet, but the builder script for 'cornell_movie_dialog' is broken
    
    `y = load_dataset('cornell_movie_dialog')['train']`
- 'Hello-SimpleAI/HC3' is broken too
    
    `z = load_dataset('Hello-SimpleAI/HC3')['train']`
- The NPS Chat Corpus is a part of NLTK, and it's completely unusable because it is a loose collection of posts and not (somewhat coherent) dialogs. 


I was left with DailyDialogue and manually-processed HC3.

By the way, these should've worked better if the data also contained prompts (like 'USER1 says '), in a manner in which the chatbot is prompted (as in `src/chatbot.py`).

## Breaking the ice
was done in a simple manner of choosing pre-defined prompts and adding them to a chatbot history. So was 'flirting' and 'committed' attitudes, I assumed that for a well-trained model (hey! some datasets have emotions labeled!) they would suffice.


## Closing remarks
Thank you for your time! I remembered how much I dislike Huggingface once you have to do something aside from most cited tasks.
