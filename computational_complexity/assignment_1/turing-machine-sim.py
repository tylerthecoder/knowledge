from __future__ import annotations
from xml.dom.minidom import parseString, Element
from typing import List, Dict, Tuple

class TransitionTable:
	# transitionLookup[fromState][read] = (toState, write, direction)
	table: Dict[str, Dict[str, Tuple[str, str, str]]] = {}

	def add_jff_xml(self, node: Element) -> None:
		def getLabel(y):
			child = node.getElementsByTagName(y)[0]
			return "_" if child.firstChild is None else child.firstChild.nodeValue

		if getLabel("from") not in self.table:
			self.table[getLabel("from")] = {}
		self.table[getLabel("from")][getLabel("read")] = (getLabel("to"), getLabel("write"), getLabel("move"))

	def get(self, fromState: str, read: str) -> Tuple[ str, str, str ]:
		return self.table[fromState][read]

class Tape:
	__leftMostPos = 0
	__rightMostPos = 0

	def __init__(self, input: str) -> None:
		self.tape = dict(enumerate(input))
		self.position = 0
		self.__leftMostPos = 0
		self.__rightMostPos = len(input) - 1

	def read(self) -> str:
		return self.tape.get(self.position, "_")

	def write(self, val: str) -> None:
		self.tape[self.position] = val

	def move(self, direction: str) -> None:
		dirNum = 0 if direction == "S" else 1 if direction == "R" else -1
		self.position += dirNum

		if self.position < self.__leftMostPos:
			self.__leftMostPos = self.position

		if self.position > self.__rightMostPos:
			self.__rightMostPos = self.position

	def __str__(self) -> str:
		return "".join(self.tape.get(i, "_") for i in range(self.__leftMostPos, self.__rightMostPos + 1))

class TM:
	@staticmethod
	def make_from_jff(file: str) -> TM:
		with open(file, "r") as f:
			dom = parseString(f.read())
			transitions = TransitionTable()
			startNode = dom.getElementsByTagName("initial")[0].parentNode.getAttribute("id")
			endNodes = [x.parentNode.getAttribute("id") for x in dom.getElementsByTagName("final")]
			for x in dom.getElementsByTagName("transition"):
				transitions.add_jff_xml(x)
			return TM(transitions, startNode, endNodes)

	def __init__(self, transitions: TransitionTable, startState: str, endStates: List[str]) -> None:
		self.transitions = transitions
		self.startState = startState
		self.endStates = endStates

	def run(self, input: str) -> None:
		tape = Tape(input)
		state = self.startState
		while state not in self.endStates:
			val = tape.read()
			toState, write, direction = self.transitions.get(state, val)
			tape.write(write)
			tape.move(direction)
			print(tape)
			state = toState

if __name__ == "__main__":
	tm = TM.make_from_jff("problem2.jff")
	tm.run("10101")
