"""

Here we have utility functions for:
tokenizing, parsing, and interpreting a simple domain-specific language (DSL).

"""

from enum import Enum
from dataclasses import dataclass

class TokenType(Enum):
    NUMBER      = 1
    LITERAL     = 2
    STRING      = 3
    OPERATOR    = 4
    SPECIAL     = 5

# console color codes
class ConsoleColor:
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    PURPLE  = '\033[95m'
    CYAN    = '\033[96m'
    END     = '\033[0m'

@dataclass
class Token:
    type: TokenType
    value: any
    line: int
    col: int


    def is_float(self):
        return self.type == TokenType.NUMBER and '.' in self.value
    
    def get_float(self):
        return float(self.value)

class TokenStream:

    """
        General purpose token stream for tokenizing a string for a C like language.
    """

    def __init__(self, string, _file_path):
        self.tokens     = []
        self.string     = string
        self.lines      = string.split('\n')
        self.pos        = 0
        self.file_path  = _file_path if _file_path else "NONE"

        line    = 0
        col     = 0
        cur_col = 0
        cur_str = ""
        idx     = 0
        def _flush_token():
            nonlocal cur_str
            nonlocal line
            nonlocal col
            nonlocal idx
            nonlocal self
            if cur_str:
                self.tokens.append(Token(TokenType.LITERAL, cur_str, line, col))
                cur_str = ""
        try:
            while idx < len(string):
                c       = string[idx]
                nc      = string[idx + 1] if idx + 1 < len(string) else None
                nnc     = string[idx + 2] if idx + 2 < len(string) else None
                nnnc    = string[idx + 3] if idx + 3 < len(string) else None
                if c == '\n':
                    line += 1
                    cur_col = 0
                    _flush_token()
                elif c == ' ' or c == '\t':
                    _flush_token()
                # skip comments
                elif c == '/' and nc == '/':
                    while nc and nc != '\n':
                        idx += 1
                        nc = string[idx + 1] if idx + 1 < len(string) else None
                
                # tripple quoted strings
                elif (c == '"' and nc == '"' and nnc == '"') or (c == "'" and nc == "'" and nnc == "'"):
                    _flush_token()
                    cur_str = ""
                    idx += 2
                    nc = string[idx + 1] if idx + 1 < len(string) else None
                    while nc and nnc and nnnc and (nc != c or nnc != c or nnnc != c):
                        cur_str += nc
                        idx     += 1
                        nc      = string[idx + 1] if idx + 1 < len(string) else None
                        nnc     = string[idx + 2] if idx + 2 < len(string) else None
                        nnnc    = string[idx + 3] if idx + 3 < len(string) else None
                    assert nc == c and nnc == c and nnnc == c, f"Expected closing tripple quote at line {line}, col {col}"
                    idx += 3
                    self.tokens.append(Token(TokenType.STRING, cur_str, line, col))
                    cur_str = ""
                # strings
                elif c == '"' or c == "'":
                    _flush_token()
                    cur_str = ""
                    while nc and nc != c:
                        cur_str += nc
                        idx     += 1
                        nc      = string[idx + 1] if idx + 1 < len(string) else None
                    assert nc == c, f"Expected closing quote at line {line}, col {col}"
                    idx += 1
                    self.tokens.append(Token(TokenType.STRING, cur_str, line, col))
                    cur_str = ""
                elif c in ['+', '-', '*', '/', '%', '>', '<', '=', '!', '&', '|', '^', '~', '?', ':', ';', ',', '.', '@', '#', '$', '`', '\\', '/', '(', ')', '{', '}', '[', ']']:
                    _flush_token()
                    self.tokens.append(Token(TokenType.SPECIAL, c, line, col))
                elif c.isdigit():
                    _flush_token()
                    cur_str += c
                    is_science = False
                    is_float = False
                    while nc and (nc.isdigit() or nc == '.' or nc == 'e' or nc == 'E' or is_science and (nc == '+' or nc == '-')):
                        if nc == 'e' or nc == 'E':
                            is_science = True
                        elif nc == '.' and not is_science:
                            is_float = True

                        cur_str += nc
                        idx += 1
                        nc = string[idx + 1] if idx + 1 < len(string) else None
                    self.tokens.append(Token(TokenType.NUMBER, cur_str, line, col))
                    cur_str = ""
                
                else:
                    if len(cur_str) == 0:
                        col = cur_col - 1
                    cur_str += c
                cur_col += 1
                idx += 1
        except Exception as e:
            self.print_error_at_current(f"Error while tokenizing: {e}")
            raise e

    def get_line(self, line):
        return self.lines[line]
    
    def print_error_at_current(self, message):
        token = self.tokens[min(self.pos, len(self.tokens) - 1)]
        # put red color on error message
        print(f"{ConsoleColor.RED}", end="")
        print(f"**************************************************")
        print(f"Error at line {token.line}, col {token.col}: {message}")
        print(f"File: {self.file_path}:{token.line + 1}")
        print(f"{self.get_line(token.line)}")
        # put cursor at col
        print(f"{'-' * token.col}^")
        print(f"**************************************************")
        print(f"{ConsoleColor.END}", end="")

    def Consume(self, string):
        n = self.next()
        return n == string

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def next(self):
        token = self.peek()
        self.pos += 1
        return token

    def eof(self):
        return self.pos >= len(self.tokens)


if __name__ == "__main__":
    
    string ="""
    // this is a comment
    x = 10;
    y : int = 20;
    t = "hello";
    x = '''-
    this is a tripple quoted string
    ''';

    Function test(x: int, y: int) -> int {
        return x + y;
    }

    test(10, 20);    

    num : f32 = 10.0e-10;
    num : f32 = 10.0e+10;
    num : f32 = 10.0e10;
    num : f32 = 10.0E10;
    num : f32 = f32(1.2);

    z"""
    tk = TokenStream(string, "test.dsl")
    print(tk.tokens)
    tk.print_error_at_current("test error")
    
    for t in tk.tokens:
        if t.is_float():
            print(f"{t.value} is a float = {t.get_float()}")

    pass