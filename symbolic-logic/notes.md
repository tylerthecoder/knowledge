# Symbolic Logic
**Professor**: Barry Ward

## Week 1

### Informal logic

All substance of typer Y conduct electricity. <br />
X conducts electricity <br />
So X is a type Y

- This is a bad argument, X could be a type Z that also conducts electricity

All substance of type Y conduct electricity. <br />
X is a substance of type Y <br />
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
- Can you just put "Its not the case that"?
- no, "tyler is cool and it is not the case that john is cool"



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







