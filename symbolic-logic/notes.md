# Symbolic Logic
**Professor**: Barry Ward

## Week 1

### Informal logic

All substance of typer Y conduct electricity. <br />
X conducts electricity<br/>
So X is a type Y

- This is a bad argument, X could be a type Z that also conducts electricity

All substance of type Y conduct electricity. <br />
X is a substance of type Y<br />
So X conducts electricity

- This is a good argument
- If the premises are true, then the conclusions has to be true (valid)


**Valid argument** -  An argument is valid iff it is impossible for all of its premises to be true and its conclusion to be false

Suppose I have an argument and all of its premises are true and its conclusion is true, must it be valid? 
- No, it could be a coincidence
    - I like pizza, so it is not raining outside

Suppose I have an argument with a false conculsion, must it be invalid? 
- No, it could be a coincidence
    - All winged creatures can fly, penguins are winged creatures, so penguins can fly


**Sound argument** - An argument is sound iff it is valid and all of its premises are true


### Formal Logic

It is not the case that both Tony and Greg will support the policy. <br />
- This can be converted to: In is not the case that Tony will support the policy and Greg will support the policy
- Not T and G

Tony will support the policy. <br />
- T

So Greg will not support the policy. <br />
- Not G


"and" and "not" is a truth function. 

"and" truth function
| p | q | p $\land$ q |
|---|---|-------------|
| T | T | T           |
| T | F | F           |
| F | T | F           |
| F | F | F           |

"not" truth function
| p | $\lnot$ p |
|---|-----------|
| T | F         |
| F | T         |


argument truth table
| T | G | (P1) $\lnot$ (T $\land$ G)  | (P2) T | (P3) $\lnot$ G  |
|---|---|-----------------------------|--------|-----------------|
| T | T | F                           | T      | F               |
| T | F | T                           | T      | T               |
| F | T | T                           | F      | F               |
| F | F | T                           | F      | T               |

No row / possiblity has the case where all the premises are true and the conclusion is false. So the argument is valid.

- This was crazy to me. I thought the proving the conculsion provably followed from the premises was much more vauge and difficult than this. Maybe because there are no "ifs"


Upercase letters stand for particular sentences. Lowercase letters stand for any sentence, it is a variable.


Ethier they are very talented or they studied very hard.

"or" Truth function
| p | q | p $\lor$ q  | p $\oplus$ q | 
|---|---|-------------|--------------|
| T | T | T           | F            |
| T | F | T           | T            |
| F | T | T           | T            |
| F | F | F           | F            |


$p \oplus q = (p \lor q) \land \lnot (p \land q)$

Truth function equivalence: Two functions have the same truth tables

Negations, 
How do you build the negation of a sentence? 
- It is raiting ( R )
- It is not raining ($\lnot$ R)
- You might think you just stick a $\lnot$ in front of the sentence, but this is not always the case
- Some people are rich ( P )
- Some people are not rich ( Q )
- $\lnot$ P $\neq$ Q
- Can you just put "It's not the case that"?
- No, "Tyler is cool, and it is not the case that john is cool"



p or q - disjuctions
p, q are disjuncts


### Conditional


If Brazil gets to the world cup final, then they will win the world cup.

First part of conditional is the antecedent, second part is the consequent. The whole sencentence is called a conditional

If B then W

Truth function
| B | W | B $\rightarrow$ W |
|---|---|-------------------|
| T | T | T                 |
| T | F | F                 |
| F | T | T                 |
| F | F | T                 |

Intuition is hard for this, but we need to think what is possible. 


The line is constituted by an infinite number of points. <br />
Each point either has length zero or the same positive length $\delta$ <br />
So all lines have either length zero or infinite length

- argument by zeno
- It is missing the premise that the length of a line is defined by the length of the sum of the consituted points, but we don't think like that. 

$\rightarrow$ is called the material conditional

If mary is in Paris, then she is in France
 - This sentence isn't really caputred by the material conditional. 


John will pass the course if and only if he passes the final
- This is a biconditional
- C iff F

Truth Table:
| C | F | C $\leftrightarrow$ F |
|---|---|------------------------|
| T | T | T                      |
| T | F | F                      |
| F | T | F                      |
| F | F | T                      |

iff is the same as $\lnot \oplus$

Called the material biconditional

Same as saying (C $\rightarrow$ F) $\land$ (F $\rightarrow$ C)
- You can check equality by checking that the truth tables are the same


Example:

Deck of cards. The backs are red or black, on the other side there is a number. 

If the back is black, then the number is even

To check, need to flip over black cards, and cards that are showing odd numbers


p | q | p $\rightarrow$ q | $\lnot$ q $\rightarrow$ $\lnot$ p |
--|---|-------------------|----------------------------------|
T | T | T                 | T                                |
T | F | F                 | F                                |
F | T | T                 | T                                |
F | F | T                 | T                                |

This table is called the contrapositive.

If Oswald hadn't shot kennedy, then someone else woudl have.
- This is a counterfactual or subjunctive conditional


If Oswald didn't shoot kennedy, then someone else did. 
- This is an indicitive conditional and is true. 

## Rules of formation

Grammatical rules or PL (propositional logic), how to make "well formed formulas" (wff)

A, B, C, D, E

Binary Connectives: $\land$, $\lor$, $\rightarrow$, $\leftrightarrow$

If p and q are wffs, then ($p \land q) is a wff
($p \lor q) is a wff. 

If we think of these recursively, if p or q is bigger than a single sentence, then it already has brackets. 

As convention, we commonly drop the outermost brackets. 


## Truth Trees

You ignore ways of making premises false or the conclusion false

Strategy, minimize branching early. Pick sentences that stack instead of branch.  

Different tree rules for each type of sentences. 

Definition: A branch closes exact if it includs lines which are p and not p. 


1. $(A \land B) \rightarrow C$

2. $\lnot A \rightarrow D$
3. $\lnot(B \rightarrow (C \lor D))$
4. $B$, 3
5. $\lnot(C \lor D)$, 3
6. $\lnot C$, 5
7. $\lnot D$, 5
8. $\lnot \lnot A$, $D$, 7


1. $A \rightarrow (B \rightarrow C)$<br/>
2. $D \rightarrow (A \lor B)$<br/>
3. $\therefore \lnot (D \rightarrow C)$
 - The conclusion is negated to check the argument is valid
4. $D$, 3
5. $\lnot C$, 3
6. $\lnot D$ --- $A\lor B$, 2
    - D$\lnot D$ closes since it is a contradiction
7. $A$ --- $B \rightarrow C$, 6
8. $\lnot A$ --- $B \rightarrow C$ --- $\lnot A$ --- $B \rightarrow C$, 1
 - $A \lnot A$ closes since it is a contradiction
 - The argument is invalid since $\lnot A$ is open

The tree test is essentially a test of consistency. 

What its doing is determining whether the initial list of sentences is consistent or not. 

Open tree = consistent<br/>
Closed tree = inconsistent

How do we use tress to test is a sentence is a tautology? 

Suppose $p$ is a tautology. Then $\lnot p$ is a contradiction. So all of its branches are closed. 

i.e. $\{\lnot p\}$ is inconsistent.

IS $(A \lor B) \rightarrow (A \land B)$ a tautology?

1. $\lnot ((A \lor B) \rightarrow (A \land B))$
2. $A \lor B$, 1
3. $\lnot (A \land B)$, 1
4. $A$ ___ $B$, 2
5. $\lnot A$ ___ $\lnot B$ ___ $\lnot A$ ___ $\lnot B$, 3

Open branches: $A \lnot B$, $\lnot A B$, Not a tautology

$p$ is logically equivalent to $q$ if and only if ($p \leftrightarrow q$) is a tautology

Write down $\lnot (p \leftrightarrow q)$ at top of the tree, and if the tree closes, then p and q are logically equivalent. An open branch will show that explicitly. 



### Tree Test Meta-Theory

Soundness of the tree test: if the test says an argument is valid, it's (always) correct. <br/>
Completeness of the tree test: if an argument is valid, the test will (eventually) say so. <br/>
Decidability of the tree test: whatever argument you provided as input, the tree test always delivers a verdict (valid or invalid) <br/>



Steps: 
1. Reform soundness in a way that I can provide it. 
2. Prove something substantive about the actual tree rules we've got (Rule soundess: not to be confused with _soundness_ i.e. the soundness of the tree test)
3. Use the rule soundess of all of our rules to prove the reformulation of soundness


Soundness of the tree test rules

If the tree closes, then the initial list is inconsistent. i.e. the premises and the negated conclusion. 

Contrapositive: If the initial list is consistent, then the tree doesn't close.

__Digression__: 

Prove $\Gamma \models p$ iff $\Gamma \cup \{\lnot p\}$ is inconsistent

i. Suppose $\Gamma \models p$. 
- There is no row in the truth table where all the member of $\Gamma$ are true and $p$ is false. 
- So there is no row where all the members of $\Gamma$ are true and $\lnot p$ is true [By the true table for logical not]
- So there is no row where all the members of $\Gamma \cup \{\lnot p\}$ are true
- So $\Gamma \cup \{\lnot p\}$ is inconsistent

ii. Suppose $\Gamma \cup \{\lnot p\}$ is inconsistent


Proof (soundness of truth trees):
Suppose the inital list is consistent
- Then, there is at least one valuation / argument of truth values that makes all the sentences in the inital list ture, call it C. 
- Since all of our tree values are rule sound, application of a tree rule to a branch of the tree in which all of the sentences are true under C, yields at least one extension of the branch, (with an added node) that is also true under C. So when our tree is completed we have at least on branch taht is true under C. 
- So the set of sentences in that branch is consistent
- So the branch cannot contain a pair of sentences of form $p$ and $\lnot p$
- So the branch is open


__Rule Soundness__:

Rule $p \lor q$ makes a branch with $p$ and $q$ open

If the input is true under andy assignment, C, then at least one of the outputs is true under C. 

There are 3 types of assignemnt that make the input $p \lor q$ true. 
- (type 1) $p$ is true and $q$ is false
- (type 2) $p$ is false and $q$ is true
- (type 3) $p$ is true and $q$ is true
Each of thesee have at least one of the branches that is true. 

You can do the same for the other rules. 

Formulation of completeness of the tree test:
If the argument is valid, then the tree test says so. 

- If the inital list is incocnsistent, then the tree closes
- If the tree is ope, then the inital list is consistent (contrapositive)

Suppose the tree is open.
- Pick an open branch
- Then we can define a truth assignment by assigning truth values that make all the uncheck lines true
- So, if they are atomic sentence, make those atomic sentence true. If they are negated atomic sentences make those atomic sentences false. 
- So under this turh assignment all the uncheck lines in that branch are ture under that truth assignment too. 
- And so on up the tree : all the lines in the tree must be true under that assignment
- So the initial list must be true under that assignment
- So the inital list is consistent.
- So if the trees open, the inital list is consistent. 


