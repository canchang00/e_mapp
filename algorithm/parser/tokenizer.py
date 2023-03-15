import algorithm.parser.lex as lex
 

tokens = ['NUMBER',
'PLUS',
'MINUS',
'TIMES',
'DIVIDE',
'LPAREN',
'RPAREN',
'LBRACE',
'RBRACE',]

 
# Regular expression rules for simple tokens
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
# t_COMMA=r','

# A regular expression rule with some action code
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)    
    return t

# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
reserved = {
    'if' : 'if',
    'else' : 'else',
    'while' : 'while',
    'parallel': 'parallel',
    'alternative': 'alternative',
    'repeat': 'repeat',
    'True': 'True',
}

tokens=tokens+list(reserved.values())

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'ID')    # Check for reserved words
    return t

t_ignore_END= r';'
t_ignore_quotation=r'\"'
t_ignore_comma=r','



lexer = lex.lex()



if __name__=="__main__":
    x=open("program.txt").read()
    lexer.input(x)
    while True:
        print(next(lexer))