# haiku verifiers environment

import re
import pyphen
import datasets as ds
import verifiers as vf

##
## constants
##

SYSTEM_PROMPT = """Haikus are special types of poems with three lines. The first line has 5 syllables, the second line has 7 syllables, and the third line has 5 syllables. Generate a haiku in response to the user's prompt. Write the final haiku enclosed in <haiku></haiku> tags."""

##
## dataset loading
##

def load_haiku_dataset(split):
    dataset = ds.load_dataset('checkai/instruction-poems', split=split)
    return dataset.map(lambda row: {
        'question': row['INSTRUCTION'],
    })

##
## utility functions
##

dic = pyphen.Pyphen(lang='en_US')

def syllable_word(word):
    return len(dic.inserted(word).split('-'))

def syllable_count(text):
    if len(text) == 0:
        return 0
    return sum(
        syllable_word(word) for word in text.split()
    )

def pad_list(a, n, v=0):
    if len(a) >= n:
        return a
    return a + [v] * (n - len(a))

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

##
## parsers and reward functions
##

def extract_haiku(text):
    matches = re.findall(r'<haiku>(.*?)</haiku>', text, re.DOTALL)
    return matches[-1].strip() if len(matches) > 0 else ''

def count_haiku(text):
    lines0 = [s.strip() for s in text.split('\n')]
    lines = [s for s in lines0 if len(s) > 0]
    return [syllable_count(line) for line in lines]

def reward_counts(count):
    s1, s2, s3, *_ = pad_list(count, 3, v=0)
    n_lines = len(count)
    penalty = (
        abs(n_lines - 3)
        + abs(s1 - 5)
        + abs(s2 - 7)
        + abs(s3 - 5)
    )
    return -(penalty / 20)

def reward_format(text, think=False):
    reward = 0
    if think:
        # if '<think>' in text: reward += 1
        if '</think>' in text: reward += 1
    if '<haiku>' in text: reward += 1
    if '</haiku>' in text: reward += 1
    total = 3 if think else 2
    return reward / total

def reward_length(text, min_length=512, max_length=1024):
    frac = (len(text) - min_length) / (max_length - min_length)
    return -clamp(frac, 0, 1)

##
## environment definition
##

def load_environment(
    num_train_examples=5000,
    num_eval_examples=500,
    use_thinking=True,
    min_length=2048,
    max_length=16384,
):
    # define reward functions
    def reward_format_function(parser, completion, **kwargs):
        text = parser.parse_answer(completion)
        return reward_format(text, think=use_thinking)
    def reward_haiku_function(parser, completion, **kwargs):
        reply = parser.parse_answer(completion)
        text = extract_haiku(reply)
        counts = count_haiku(text)
        return reward_counts(counts)
    def reward_length_function(parser, completion, **kwargs):
        reply = parser.parse_answer(completion)
        return reward_length(reply, min_length=min_length, max_length=max_length)

    # load training data
    dataset = load_haiku_dataset('train')
    train_dataset = dataset.select(range(num_train_examples))
    eval_dataset = dataset.select(
        range(num_train_examples, num_train_examples + num_eval_examples)
    )

    # thinking parser
    parser = vf.Parser()

    # set up haiku reward rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            reward_format_function,
            reward_haiku_function,
            reward_length_function,
        ],
        weights=[1.0, 1.0, 1.0],
    )

    # set up environment
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
