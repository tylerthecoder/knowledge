
# Assignment 1

## Section A

### Question 1

Determine whether each of the following arguments is valid:

(a) $A \rightarrow B, \lnot C \rightarrow \lnot B \therefore A \rightarrow C$

Truth Table

| A | B | C | $A \rightarrow B$ | $\lnot C \rightarrow \lnot B$ | $A \rightarrow C$ |
|---|---|---|-------------------|-------------------------|-------------------|
| T | T | T | T                 | T                       | T                 |
| T | T | F | T                 | F                       | F                 |
| T | F | T | F                 | T                       | T                 |
| T | F | F | F                 | T                       | F                 |
| F | T | T | T                 | T                       | T                 |
| F | T | F | T                 | F                       | T                 |
| F | F | T | T                 | T                       | T                 |
| F | F | F | T                 | T                       | T                 |

This argument is valid. When the premises are true, the conclusion is also true.

(b) $A \lor B, A \rightarrow C, D \rightarrow B \therefore C \lor D$

Truth Table

| A | B | C | D | $A \lor B$ | $A \rightarrow C$ | $D \rightarrow B$ | $C \lor D$ |
|---|---|---|---|------------|--------------------|--------------------|------------|
| T | T | T | T | T          | T                  | T                  | T          |
| T | T | T | F | T          | T                  | T                  | T          |
| T | T | F | T | T          | F                  | T                  | T          |
| T | T | F | F | T          | F                  | T                  | F          |
| T | F | T | T | T          | T                  | F                  | T          |
| T | F | T | F | T          | T                  | T                  | T          |
| T | F | F | T | T          | F                  | F                  | T          |
| T | F | F | F | T          | F                  | T                  | F          |
| F | T | T | T | T          | T                  | T                  | T          |
| F | T | T | F | T          | T                  | T                  | T          |
| F | T | F | T | T          | T                  | T                  | T          |
| F | T | F | F | T          | T                  | T                  | T          |
| F | F | T | T | F          | T                  | T                  | T          |
| F | F | T | F | F          | T                  | T                  | T          |
| F | F | F | T | F          | T                  | T                  | T          |
| F | F | F | F | F          | T                  | T                  | F          |

This argument is valid. When the premises are true, the conclusion is also true.

### Question 2

For each of the following sentences, determine if it is a tautology, a contradiction, or neither (i.e., contingent).

(a) $A \rightarrow (A \lor B)$

Truth Table

| A | B | $A \lor B$ | $A \rightarrow (A \lor B)$ |
|---|---|------------|-----------------------------|
| T | T | T          | T                           |
| T | F | T          | T                           |
| F | T | T          | T                           |
| F | F | F          | T                           |

This is a tautology. The conclusion is always true.


(b) $\lnot(\lnot A \lor \lnot B) \leftrightarrow (A \land B)$

Truth Table

| A | B | $\lnot (\lnot A \lor \lnot B)$ | $A \land B$ | $\lnot (\lnot A \lor \lnot B) \leftrightarrow (A \land B)$ |
|---|---|--------------------------------|-------------|------------------------------------------------------------|
| T | T | T                              | T           | T                                                          |
| T | F | F                              | F           | T                                                          |
| F | T | F                              | F           | T                                                          |
| F | F | F                              | F           | T                                                          |

This is a tautology. The conclusion is always true.


### Question 3

Determine whether the following pairs of sentences are logically equivalent.

(a) $(A \land \lnot B) \lor (\lnot A \land B)$ and $A \leftrightarrow \lnot B$

Truth Table

| $A$ | $B$ | $(A \land \lnot B) \lor (\lnot A \land B)$ | $A \leftrightarrow \lnot B$ |
|-----|-----|--------------------------------------------|------------------------------|
| T   | T   | F                                          | F                            |
| T   | F   | T                                          | T                            |
| F   | T   | T                                          | T                            |
| F   | F   | F                                          | F                            |


These two sentences are logically equivalent. They have the same truth values for all possible combinations of truth values for $A$ and $B$.

(b) $A \lor (B \land C)$ and $(A \lor B) \land (A \lor C)$

Truth Table

| $A$ | $B$ | $C$ | $A \lor (B \land C)$ | $(A \lor B) \land (A \lor C)$ |
|-----|-----|-----|----------------------|-------------------------------|
| T   | T   | T   | T                    | T                             |
| T   | T   | F   | T                    | T                             |
| T   | F   | T   | T                    | T                             |
| T   | F   | F   | T                    | T                             |
| F   | T   | T   | T                    | T                             |
| F   | T   | F   | F                    | F                             |
| F   | F   | T   | F                    | F                             |
| F   | F   | F   | F                    | F                             |

These two sentences are logically equivalent. They have the same truth values for all possible combinations of truth values for $A$, $B$, and $C$.


### Question 4


Translate the following sentences into our formal language, Propositional Logic

1. Not only will Germany win the European Championships, but they will win the World Cup as well. 

E = Germany will win the European Championships

W = Germany will win the World Cup

$E \land W$


2. Either we buy the groceries, or we visit your relatives, but not both.

G = We buy the groceries

R = We visit your relatives

$(G \lor R) \land \lnot (G \land R)$


3. If Kite misses his tap-in and the spectators groan loudly, then Watson's concentration will be disturbed. 

K = Kite misses his tap-in

S = The spectators groan loudly

W = Watson's concentration will be disturbed

$(K \land S) \rightarrow W$

4. I don't remember falling down, but if I did, then I must have been sleepwalking. 

R = I remember falling down

F = I fell down

S = I was sleepwalking

$\lnot R \land (F \rightarrow S)$


5. John did not go to the store, but Kate did. 

J = John went to the store

K = Kate went to the store

$\lnot J \land K$

6. At least one of John and Kate went to the store. 

J = John went to the store

K = Kate went to the store

$J \lor K$

7. Neither John nor Kate went to the store, but Peter did. 

J = John went to the store

K = Kate went to the store

P = Peter went to the store

$\lnot (J \lor K) \land P$


8. Some cities have skyscrapers, but not all of them do. 

C = A city has a skyscraper

A = All cities have skyscrapers

$C \land \lnot A$


9. I drink if, and only if, I am thirsty. 


D = I drink

T = I am thirsty

$D \leftrightarrow T$


10. I drink only if I am thirsty 

D = I drink

T = I am thirsty

$D \rightarrow T$

### Question 5

Translate the following argument into our formal language, Propositional Logic, and 
determine whether it is valid using a truth-table.  Explicitly state the English 
sentence corresponding to each sentence letter. 
 
If Sartre is an existentialist, he isn’t rational. If he’s not rational, he’s an existentialist.  
So, he’s not an existentialist if, and only if, he is rational.


E = Sartre is an existentialist

R = Sartre is rational

$E \rightarrow \lnot R$

$\lnot R \rightarrow E$

$\therefore E \leftrightarrow R$

Truth Table

| $E$ | $R$ | $E \rightarrow \lnot R$ | $\lnot R \rightarrow E$ | $E \leftrightarrow R$ |
|-----|-----|-------------------------|--------------------------|------------------------|
| T   | T   | F                       | F                        | T                      |
| T   | F   | T                       | T                        | F                      |
| F   | T   | T                       | T                        | F                      |
| F   | F   | T                       | T                        | T                      |

This argument is not valid. There are cases where the premises are true, but the conclusion is false. Namely, whenever just one of the sentences is false. 


### Question 6

True or False?  Where true, provide a proof.  Where false, provide a counterexample 
i.e., a particular case that demonstrates that the claim is false.  All counterexamples 
should be in our formal language (i.e. not arguments or sentences in English).  
 
1. If a sentence is not a contradiction, then its negation must be one. 

This is not true. Here is a counterexample:

$A \lor B$

$\lnot (A \lor B)$

Truth Table

| $A$ | $B$ | $A \lor B$ | $\lnot (A \lor B)$ |
|-----|-----|------------|--------------------|
| T   | T   | T          | F                  |
| T   | F   | T          | F                  |
| F   | T   | T          | F                  |
| F   | F   | F          | T                  |

The sentence $A \lor B$ is not a contradiction, but its negation is not either.
 
2. If a sentence is contingent (i.e., neither a tautology nor a contradiction), then its 
negation is also. 

This is true. 

Let $A$ be some sentence that is contingent. This means that the truth values for $A$ are both $T$ and $F$. If we flip all of these by negating $A$ then they still consistent of $T$ and $F$ values. Thus the claim is true. 




