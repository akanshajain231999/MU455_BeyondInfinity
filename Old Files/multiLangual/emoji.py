from emoji import UNICODE_EMOJI
from emosent import get_emoji_sentiment_rank

def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)

print(extract_emojis('my country is my first love ❤'))
get_emoji_sentiment_rank('❤')