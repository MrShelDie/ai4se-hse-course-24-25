import re

PR_PATTERNS = {
    ' fuck ':
        [
            r'(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            r'(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            r' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k',
            r'f u u c',
            r'(f)(c|[^a-z ])(u|[^a-z ])(k)',
            r'f\*',
            r'feck ',
            r' fux ',
            r'f\*\*',
            r'f\-ing',
            r'f\.u\.',
            r'f###',
            r' fu ',
            r'f@ck',
            r'f u c k',
            r'f uck',
            r'f ck',
            r'f[u]+ck'

        ],

    ' crap ':
        [
            r' (c)(r|[^a-z0-9 ])(a|[^a-z0-9 ])(p|[^a-z0-9 ])([^ ])*',
            r' (c)([^a-z]*)(r)([^a-z]*)(a)([^a-z]*)(p)',
            r' c[!@#\$%\^\&\*]*r[!@#\$%\^&\*]*p',
            r'cr@p',
            r' c r a p',

        ],

    ' haha ':
        [
            r'ha\*\*\*ha',
        ],

    ' ass ':
        [
            r'[^a-z]ass ',
            r'[^a-z]azz ',
            r'arrse',
            r' arse ',
            r'@\$\$',
            r'[^a-z]anus',
            r' a\*s\*s',
            r'[^a-z]ass[^a-z ]',
            r'a[@#\$%\^&\*][@#\$%\^&\*]',
            r'[^a-z]anal ',
            r'a s s'
        ],

    ' ass hole ':
        [
            r' a[s|z]*wipe',
            r'a[s|z]*[w]*h[o|0]+[l]*e',
            r'@\$\$hole'
        ],

    ' bitch ':
        [
            r'bitches',
            r' b[w]*i[t]*ch',
            r' b!tch',
            r' bi\+ch',
            r' b!\+ch',
            r' (b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            r' biatch',
            r' bi\*\*h',
            r' bytch',
            r'b i t c h'
        ],

    ' bastard ':
        [
            r'ba[s|z]+t[e|a]+rd'
        ],

    ' transgender':
        [
            r'transgender'
        ],

    ' gay ':
        [
            r'gay',
            r'homo'
        ],

    ' cock ':
        [
            r'[^a-z]cock[., ]',
            r'c0ck',
            r'[^a-z]cok ',
            r'c0k',
            r'[^a-z]cok[^aeiou]',
            r' cawk',
            r'(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)',
            r'c o c k'
        ],

    ' dick ':
        [
            r' dick[^aeiou]',
            r'd i c k'
        ],

    ' suck ':
        [
            r'sucker',
            r'(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)',
            r'sucks',
            r'5uck',
            r's u c k'
        ],

    ' cunt ':
        [
            r'cunt',
            r'c u n t'
        ],

    ' bull shit ':
        [
            r'bullsh\*t',
            r'bull\$hit',
            r'bull sh.t'
        ],

    ' jerk ':
        [
            r'jerk'
        ],

    ' idiot ':
        [
            r'i[d]+io[t]+',
            r'(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)',
            r'idiots',
            r'i d i o t'
        ],

    ' dumb ':
        [
            r'(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
        ],

    ' shit ':
        [
            r'shitty',
            r'(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)',
            r'shite',
            r'\$hit',
            r's h i t',
            r'sh\*tty',
            r'sh\*ty',
            r'sh\*t'
        ],

    ' shit hole ':
        [
            r'shythole',
            r'sh.thole'
        ],

    ' retard ':
        [
            r'returd',
            r'retad',
            r'retard',
            r'wiktard',
            r'wikitud'
        ],

    ' rape ':
        [
            r'raped'
        ],

    ' dumb ass':
        [
            r'dumbass',
            r'dubass'
        ],

    ' ass head':
        [
            r'butthead'
        ],

    ' sex ':
        [
            r'sexy',
            r's3x',
            r'sexuality'
        ],

    ' nigger ':
        [
            r'nigger',
            r'ni[g]+a',
            r' nigr ',
            r'negrito',
            r'niguh',
            r'n3gr',
            r'n i g g e r'
        ],

    ' shut the fuck up':
        [
            r' stfu',
            r'^stfu'
        ],

    ' for your fucking information':
        [
            r' fyfi',
            r'^fyfi'
        ],
    ' get the fuck off':
        [
            r'gtfo',
            r'^gtfo'
        ],

    ' oh my fucking god ':
        [
            r' omfg',
            r'^omfg'
        ],

    ' what the hell ':
        [
            r' wth',
            r'^wth'
        ],

    ' what the fuck ':
        [
            r' wtf',
            r'^wtf'
        ],
    ' son of bitch ':
        [
            r' sob ',
            r'^sob '
        ],

    ' pussy ':
        [
            r'pussy[^c]',
            r'pusy',
            r'pussi[^l]',
            r'pusses',
            r'(p)(u|[^a-z0-9 ])(s|[^a-z0-9 ])(s|[^a-z0-9 ])(y)'
        ],

    ' faggot ':
        [
            r'faggot',
            r' fa[g]+[s]*[^a-z ]',
            r'fagot',
            r'f a g g o t',
            r'faggit',
            r'(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)',
            r'fau[g]+ot',
            'fae[g]+ot',
        ],

    ' mother fucker':
        [
            r' motha f',
            r' mother f',
            r'motherucker',
            r' mofo',
            r' mf ',
        ],

    ' whore ':
        [
            r'wh\*\*\*',
            r'w h o r e'
        ]
}


def replace_profanity(text: str) -> str:
    """
    Replace profanity in a given text with a standardized string.

    This function iterates over all the profanity patterns and replaces any
    matches in the given text with the standardized string for that type of
    profanity. The replacement string is designed to be easily machine-readable
    and human-readable.

    Parameters
    ----------
    text : str
        The text to process.

    Returns
    -------
    str
        The text with all profanity replaced with standardized strings.
    """
    for target, patterns in PR_PATTERNS.items():
        for pat in patterns:
            text = re.sub(pat, target, text)

    return text

