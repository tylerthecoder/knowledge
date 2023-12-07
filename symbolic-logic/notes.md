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



## Quantifiers

Existential quantifier: $\exists x$
- The tree rule replaces the existential quantifier with a new name that is not in the tree yet. 
Universal quantifier: $\forall x$
- The rule is never checked off, because it applies to any instance of the variable. 
- The tree rule for the Universal quantifier is rule sound

With these rules, the tree test is still sound and complete.




## Chapter 4 (Multiple Generality)

So far we have only had one variable, x. Now we allow multiple variables. 

Why? This allows us to express much more complicated sentences formally


Domain = people
r = Romeo, j = Juliet, Lxy = x loves y

"Everyone loves themselves": $\forall x Lxx$

"Romeo loves everyone": $\forall x Lrx$

"Everyone loves Romeo": $\forall x Lxr$

"Juliet loves someone": $\exists x Ljx$

"Everyone loves someone": $\forall x \exists y Lxy$

"Somebody loves everyone": $\exists x \forall y Lxy$


Formation trees: (Scope of quantifiers) <br />

Quasi-English: (Leave the quantifiers in the sentence) <br />

When you build up a sentence with a formation tree: The line where you introduce a quantifier can often be best understood by reference to the previous line
- This means you can't put quantifiers in between others.
- Can't go from $\forall x Lxb$ to $\forall x \exists y Lxy$

Everyone loves someone != Someone is loved by everyone

$\forall x ( \exists y Lxy \rightarrow \exists yLyx)$
- Everyone who loves someone is loved by someone. 
- Everyone who loves is loved

What is a lover? 

$a$ is a lover if $\exists y Lay$

All love all lovers = For every x, if x is a lover, then everyone loves x. 
$\forall x (\exists yLxy \rightarrow \forall yLyx)$



__Infinite Trees__:

$\forall x \exists y Lxy$, $\therefore Laa$

1. $\forall x \exists y Lxy$
2. $lnot Laa$
3. $\exists y Lay$, 1
4. Lab, 3
5. $\exists y Lby$, 1
6. Continue forever

The domain is $\{a, b, c, d, ...\}$
$a \rightarrow 1, b \rightarrow 2, ...$
$L = \{(a, b), (b, c), (c, d), ...\}$


So infinite tree exist. 
The tree test will always work for valid arguments. (Sound and complete)
But, there are some invalid arguments that make the tree test go on forever.

In general, having a decision procedure for validity is uncomputable. 

### Completeness of the tree test as a test for validity

If the argument is invalid then the tree test says so. 

The tree is either finite or open or infinite then the initial list is consistent 


## Identity

Qualitative Identity: Distinct objects with the same properties 
Numerical identity: two names same object
- This is what we care about for logic

Identity Symbol: $=$ 
In terms of syntax $=$ function like a 2-place predicate. i.e., in a WOFF it has to be paired with exactly two names. 

Logical truths about identity. 
- $\forall x x=x$
- $\forall x \forall y ((Px ^ x = y) \rightarrow Py)$

New tree rules
- $\lnot a = a$, then close the tree
- If $a = b$, then you can make a new line where you replace a with b


Sherlock Holmes has a hat. $Ha$
Sherlock Holmes exists. $\exists x x = a$

Problem: When we translate a name in English / natural languae, we presuppose that the object exists. 


The sentence "Does Sherlock Holmes exists?"


The cat on the mat is black 

Definite description construction (Russel made this): 
= "There is at least one ___ and everything that satisfies ___ is identical with it"

$\exists x ((Cx \land Mx) \land \forall y ((Cy \land My) \rightarrow y = x)) \land ...)$

Russell part II: 
Names are all really definite descriptions.
i) seems to dispose of the Sherlock Holmes problem, naming someone doesn't loggicaly commit us to their existence. 
ii) seems to give an account of how to refer to someone we aren't acquainted with. 

Numerical identify is an equivalence relation. 
- Reflexivity: Everything bears this relation to itself
- Symmetry: if a bears R to b, then b bears R to a. 
- Transitivity: If a bears R to b and b bears R to C, then a bears R to c. 

Tree rules for identity allow you to replace instance of a with b if $a = b$

$\lnot b = a$ means $\lnot(a = b)$
You can't negate names. "Not Tyler" does not make sense. 

There are at least 2 Ps:
$\exists x \exists y (Px \land Py \land \lnot x = y)$

There are at most 2 Ps:
$\lnot \exists x \exists y \exists z (Px \land Py \land Pz \land \lnot x = y \land \lnot x = z \land \lnot y = z)$

There are exactly 2 Ps:
There are at least 2 Ps and there are at most 2 Ps

$ \exist x \exist y (Px \land Py \land \forall x (Pz \rightarrow (z = x \lor z = y)))

This is numbers

Achilles loathes Hector and only Heactor
$ Lah \land \forall x(Lax \rightarrow x = h) $


Achilles is feared by all Trojans who fear no other Greeks. 

i.e. All Trojans who fear no other Greeks fear Achilles.

$\forall x ((Tx \land \lnot \exists y ( Gy \land Fxy \land \lnot (y = a) )) \rightarrow Fxa)$


$\lnot \exists ( \phi(x) \land \sigma(x))$ = $\forall x ( \phi(x) \rightarrow \lnot \sigma(x))$


Achilles is feared by the (one) traojan who fears no other Greek. 

i.e. The one Trojan who fears no other Greek fears achilles


## Functions

A **function** is something we can apply to a name and yield another name. 

The father of John's father is tall

j = John, fj = The father of John, ffj = The father of the father of John

All functions are single-valued (i.e. one output)

The function must be defined over the entirety of the domain. 

A 1 place function will be a set of ordered pairs

1. $\forall x (Mx \rightarrow \forall y (x \neq Fy))$
2. $\exists x Mx$
3. Mfa, 2, EI with a new name
4. Mfa \rightarrow \forall y(fa \neq fy), UI with x = fa

We can't use EI with a function name. We don't know if there is some name that maps to the input


### Dealing with infinity

We are really stuck with infinite trees now


The first time you UI, you can only use name without function symbols. Next time around you do a single function symbol, then two, etc... This helps deal with infinite trees



## Robinson arithmetic
An axiomatizion of arithmetic. The counting numbers plus basic operations (add, multiply)

A set of basic assumptions regarding the subject in question. 

Prove all and only the true claims of that subject matter. 

Axioms
1. $\forall x \forall y (x \neq y \rightarrow Sx \neq Sy )$
2. $\forall x 0 \neq Sx$
3. $\forall x (x \nq 0 \rightarrow \exists y (x = Sy))$

S is a function is the "successor" functions

There are four other axioms that define addition and multiplication

4. $\forall (x + 0 = x)$
4. $\forall x \forall y (x + Sy = S(x + y))$

We can get 1, 2, 3, 4 by taking the successor of numbers. 

This looks good, but it doesn't axiomatise the natural numbers. 
All the truths of the natural numbers are not provable in Robinson arithmetic

We can show that Q doesn't "Pick out" the natural numbers by giving a non standard model of arithmetic that satisfies those axioms. 

Godel = there is no finite axiomatization that captures all and only all truths of arithmetic. 



## Computing Machines

Types of computing machines:
- Turing Machines
- Recursive functions (lambda calculus)
- Register machines


Register machine:
- There is a set of registers and operations that you can perform. You draw lines from states and leaving from different directions depending on the value in the register. 

Church Turning Thesis: All effectively computable functions are Turing computable. 

Lossless Adder: Compute the sum of two number and Im going to leave it that the registers that contain the input value shave those values at the end of the calculation. 
Input values: in A and B registers and then compute the sum of those values

Halting problem:
Can we construct a program that does the following: 
Given a register machine program and the input, tell me if that program will ever finish. 

We can prove by contradictions. We show that you can name each program with a number. So our program would take two numbers (the program and the input). 

We can enumerate all the register machines. 

### Halting problem

We can give each register machine a number. 
So we input the name of a register machine that can hypothetical machine that computes the halting function. 

$h(m,n)$ = 1 if f_m(n) is defined and halts, 0 otherwise

Self halting problem is when m = n

This is the halting problem for all machines. There is a subset of machines that you can solve the halting problem for. 

Proof: 
Suppose there was a machine that solved the self halting problem. 

You tack on a circuit that halts if the machine doesn't halt and vice versa. Call new machine g. 

What happens if I feed g its own name? It halts when g doesn't halt, this is a contradiction. 

We can use the fact that you can't solve the halting problem to show that there is no decision procedure for validity. I.e. if the argument is valid, your procedure tells you so and if the argument is invalid, your procedure tells you so. 

We prove this by showing if there was a decision procedure for validity, we could solve the halting problem. 

If we have a decision procedure for validity, then we can mechanically assess the validity of 

We prove both that
- If the argument is valid, then the machine halts
- If the machine halts, then the argument is valid


## Modal logic

We want to expand the expressive capabilities of our formal languages
- Necessary 
- Possible
- Impossible

Why do we talk about things that are not possible / non-actual 

Let's try to formalize this. 

Modal propositional logic
- It is possible that p: $\Diamond p$
- It is necessary that p: $\Box p$

Question: are $\Box$ and $\Diamond$ truth functional?
- No, these do not have truth tables

$\Box (2+2=4) = T$

$\Box$(I am over 10 years old) $= F$

Helpful suggestion (from Leibniz):
- Necessity = truth in all possible worlds
- Possibility = truth in at least one possible world. 

Kripke
- Put this on a formal footing

We need to define what a possible world is

Modal Semantics
- The formal articulation of the possible worlds idea
- We added the new two new operators to PL

An interpretation is an ordered triple $<W,R,V>$. 
- $W$ is a non-empty set of objects, the set of possible worlds. 
- $R$ is a binary relation (The accessibility function, tells you which worlds are accessible from which) 
- $V$ to a two-place function that assigns a truth value to each sentence relative to each world in the set W. 

We can think of a world as a set of sentences that are true


This needs to hold:
- $V_w(\lnot p) = T iff V_w(p) = 0$
- $V_w(\Diamond p) = T \text{ iff } \exists w' \in W \land wRw' \land V_{w'}(p) = T$
    - P must be true in world w' that is related to w
- $V_w(\Box p) = T \text { iff } \forall w' \in W, wRw', V_{w'}(p) = T$

Note if no worlds are accessible from w, then $\Box p = 1$ for all p.

We want: $\lnot \Diamond p = \Box \lnot p$


