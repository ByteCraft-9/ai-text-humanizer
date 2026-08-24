"""Frozen word lists used by the feature extractor.

Generated once and committed rather than downloaded: the feature vector must
be identical during training and at inference, and a list that can shift
between the two is a silent accuracy bug. Keep this file stable — changing it
invalidates every trained checkpoint.
"""

from __future__ import annotations

COMMON_WORDS: frozenset[str] = frozenset(
    (
    'a', 'able', 'about', 'above', 'activity', 'add', 'after', 'again', 'against',
    'age', 'agree', 'air', 'all', 'allow', 'also', 'am', 'an', 'and', 'another', 'any',
    'appear', 'are', 'art', 'as', 'ask', 'at', 'back', 'bad', 'be', 'because', 'been',
    'before', 'begin', 'being', 'believe', 'below', 'between', 'big', 'body', 'book',
    'both', 'boy', 'break', 'bring', 'build', 'building', 'business', 'but', 'buy',
    'by', 'call', 'came', 'can', 'car', 'care', 'carry', 'case', 'catch', 'cause',
    'change', 'child', 'children', 'choose', 'city', 'class', 'college', 'come',
    'coming', 'community', 'company', 'consider', 'continue', 'control', 'could',
    'country', 'court', 'cover', 'create', 'customer', 'cut', 'day', 'death', 'decide',
    'decision', 'develop', 'development', 'did', 'die', 'different', 'do', 'does',
    'doing', 'done', 'door', 'down', 'draw', 'drug', 'during', 'each', 'early', 'eat',
    'education', 'effect', 'effort', 'either', 'end', 'even', 'event', 'every',
    'expect', 'experience', 'explain', 'eye', 'face', 'fact', 'fall', 'family',
    'father', 'feel', 'few', 'field', 'figure', 'find', 'first', 'floor', 'follow',
    'foot', 'for', 'force', 'form', 'friend', 'from', 'further', 'game', 'get', 'girl',
    'give', 'go', 'goal', 'going', 'gone', 'good', 'government', 'great', 'ground',
    'group', 'grow', 'guy', 'had', 'hand', 'happen', 'has', 'have', 'he', 'head',
    'health', 'hear', 'heart', 'help', 'her', 'here', 'high', 'him', 'his', 'history',
    'hit', 'hope', 'hour', 'house', 'how', 'i', 'idea', 'if', 'important', 'in',
    'include', 'information', 'interest', 'into', 'is', 'issue', 'it', 'its', 'job',
    'just', 'keep', 'kid', 'kill', 'kind', 'know', 'large', 'last', 'late', 'law',
    'lead', 'leader', 'learn', 'least', 'leave', 'left', 'less', 'let', 'letter',
    'level', 'life', 'light', 'like', 'line', 'listen', 'little', 'live', 'long',
    'look', 'lose', 'love', 'low', 'made', 'make', 'man', 'many', 'market', 'me',
    'meet', 'member', 'men', 'mind', 'minute', 'moment', 'more', 'morning', 'most',
    'move', 'much', 'music', 'my', 'name', 'nation', 'need', 'neither', 'new', 'next',
    'no', 'not', 'now', 'number', 'of', 'off', 'offer', 'office', 'old', 'on', 'once',
    'one', 'only', 'open', 'or', 'other', 'our', 'out', 'over', 'own', 'page', 'paper',
    'parent', 'part', 'party', 'pass', 'pay', 'people', 'person', 'place', 'plan',
    'play', 'point', 'police', 'policy', 'power', 'president', 'price', 'problem',
    'process', 'produce', 'program', 'project', 'property', 'provide', 'public',
    'pull', 'question', 'raise', 'rate', 'reach', 'read', 'realize', 'reason',
    'receive', 'record', 'remain', 'remember', 'report', 'require', 'research',
    'result', 'return', 'right', 'role', 'run', 'said', 'same', 'say', 'school', 'see',
    'seem', 'sell', 'send', 'sense', 'serve', 'service', 'set', 'she', 'short', 'show',
    'side', 'sit', 'small', 'so', 'some', 'son', 'song', 'south', 'space', 'speak',
    'spend', 'stand', 'star', 'start', 'state', 'stay', 'stop', 'street', 'student',
    'study', 'such', 'suggest', 'support', 'system', 'table', 'take', 'talk',
    'teacher', 'team', 'tell', 'than', 'thank', 'that', 'the', 'their', 'them', 'then',
    'there', 'these', 'they', 'thing', 'think', 'this', 'those', 'through', 'time',
    'to', 'try', 'turn', 'two', 'under', 'understand', 'up', 'us', 'use', 'very',
    'voice', 'wait', 'walk', 'want', 'war', 'was', 'watch', 'water', 'way', 'we',
    'week', 'well', 'went', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
    'why', 'wife', 'will', 'win', 'with', 'woman', 'women', 'word', 'work', 'world',
    'would', 'write', 'year', 'you', 'young', 'your',
    )
)
"""High-frequency English words. Stands in for the Dale-Chall easy-word list
and defines the complement used by `rare_word_ratio`."""

FUNCTION_WORDS: frozenset[str] = frozenset(
    (
    'a', 'about', 'above', 'after', 'against', 'all', 'also', 'am', 'among', 'an',
    'and', 'another', 'any', 'are', 'around', 'as', 'at', 'be', 'been', 'before',
    'behind', 'being', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond',
    'both', 'but', 'by', 'can', 'could', 'did', 'do', 'does', 'doing', 'down',
    'during', 'each', 'either', 'every', 'few', 'for', 'from', 'had', 'has', 'have',
    'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'inside', 'into', 'is', 'it', 'its', 'itself', 'just', 'least',
    'less', 'many', 'may', 'me', 'might', 'mine', 'more', 'most', 'much', 'must', 'my',
    'myself', 'near', 'neither', 'no', 'nor', 'not', 'of', 'off', 'on', 'only', 'onto',
    'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over',
    'own', 'same', 'several', 'shall', 'she', 'should', 'since', 'so', 'some', 'such',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'these', 'they', 'this', 'those', 'through', 'till', 'to', 'too', 'under', 'until',
    'up', 'upon', 'us', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'who', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    )
)
"""Closed-class words. Their complement is the content-word set used by
lexical density and the coherence cosine."""

SUBORDINATORS: frozenset[str] = frozenset(
    (
    'after', 'albeit', 'although', 'as', 'because', 'before', 'if', 'once', 'provided',
    'since', 'so', 'that', 'though', 'unless', 'until', 'when', 'whenever', 'where',
    'whereas', 'wherever', 'whether', 'while', 'why',
    )
)
"""Subordinating conjunctions, used as the clause-depth proxy."""

DISCOURSE_MARKERS: frozenset[str] = frozenset(
    (
    'accordingly', 'additionally', 'alternatively', 'as a result', 'besides',
    'by contrast', 'consequently', 'conversely', 'essentially', 'finally', 'first',
    'firstly', 'for example', 'for instance', 'fundamentally', 'furthermore', 'hence',
    'however', 'importantly', 'in addition', 'in conclusion', 'in contrast',
    'in essence', 'in particular', 'in summary', 'indeed', 'instead',
    'it is important to note', 'it is worth noting', 'it should be noted', 'lastly',
    'likewise', 'meanwhile', 'moreover', 'nevertheless', 'nonetheless', 'notably',
    'on the other hand', 'overall', 'particularly', 'second', 'secondly', 'similarly',
    'specifically', 'subsequently', 'that said', 'therefore', 'third', 'thirdly',
    'thus', 'ultimately',
    )
)
"""Connectives that LLM prose over-produces. Multi-word entries are matched
against the joined lower-cased text, single words against the token list."""
