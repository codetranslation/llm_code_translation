from enum import Enum
import constants


def get_code_smell_analyzer_by_pl(pl):
    if pl == constants.java:
        return CodesmellAnalyzer.PMD
    elif pl == constants.python:
        return CodesmellAnalyzer.PyLint
    elif pl == constants.scala:
        return CodesmellAnalyzer.ScalaStyle
    elif pl == constants.rust:
        return CodesmellAnalyzer.Clippy


class CodesmellAnalyzer(Enum):
    PyLint = 1
    PMD = 2
    ESLint = 3
    ScalaStyle = 4
    Clippy = 5


class Codesmell:
    def __init__(self, code: str, message: str, line: int, cs_type: str, analyzer: CodesmellAnalyzer, programming_language: str):
        self.code = code
        self.message = message
        self.line = line
        self.type = cs_type
        self.analyzer = analyzer
        self.programming_language = programming_language

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Codesmell):
            return self.code == other.code and self.programming_language == other.programming_language
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple([self.code, self.programming_language]))
