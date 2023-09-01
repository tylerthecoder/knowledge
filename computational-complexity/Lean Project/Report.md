
<div style="margin-bottom: 200px"></div>

<h1 align="center">Complexity Theory Project</h1>
<p align="center">Tyler Tracy</p>
<p align="center">University of Arkansas</p><div style="page-break-after: always;"></div>













## Introduction

My project was to analyze the formalization of computational models and computational complexity in Lean by examining a GitHub repository with a formalization of Merge Sort and a proof of its complexity in the hierarchical memory computational model. Then I contribute to the project by adding definitions for asymptotically complexity and formalizing the complexity class P and proving that the merge sort algorithm is a member of the complexity class P. 

Formalizing mathematical and computer science topics has been a major focus for computer science and mathematics since their conception. One of the challenges of formalizing these concepts is the tedious and error-prone task of checking formalizations. To address this challenge, there has been a push to write concepts in computer-provable formalization languages. Lean is one such language that uses type theory to write mathematical proofs in a way that a computer can check by “compiling” the proof, ensuring that the formalization is complete and correct without relying on human checking. 

Lean relies heavily on dependent type theory to have an expressive language to define abstract structures and proofs about those structures. The proofs take a form where there is some “goal” that you are proving, and each line of the proof alters that goal somehow until the last goal is achieved. There are other computer theorems proves like Coq or Isabel. They have been around for a bit longer, and they are a more verbose in their approach and aren't as powerful as Lean in certain regards. 

Although there is a standard math library with definitions and proofs for most mathematical structures like natural numbers, sets and groups, there is currently no definition for computation complexity classes in the standard library. There are scattered definition in Coq and Isabel but nothing major either. I've searched around at different approaches at attempting to formalize these concepts and decided that Andrew Carter's attempt to formalize complexity in Lean was the most promising approach due to its recent progress and improved methodology. Thus I will be analyzing his approach.


## Technical overview of repository

Now I will give an overview on Andrew Carter's progress on formalizing complexity in Lean. My goal is to give a technical overview of the theorems and to highlight important structures or type definitions, but not to explain every line of the codebase. The basic idea of the repository is to formalize a generic computational model that can be instantiated in different forms and used to write proofs about programs in that computational mode. There are two computational models that the repository creates; the hierarchical memory model and the lambda calculus model. I will be focusing on the hierarchical memory model. The model defines a way to write programs and prove the precise amount of operations a program will make on a certain input.  The main result of the repository is to prove an upper bound on the number of operations Merge Sort makes in this computational model. 

### Computational model

We will first go over the computational model structure, which is defined below. The structure is a snippet of Lean code taken from the GitHub repository. 

```lean
structure model :=
  (Program: Type)
  (Data: Type)
  (Result: Type)
  (Cost: Type)
  (data_equiv: setoid Data)
  (result_equiv: setoid Result)
  (cost_lt: preorder Cost)
  (cost_add: has_add Cost)
  (apply: Program → Data → Program)
  (has_cost: Program → Cost → Prop)
  (result: Π {p c}, (has_cost p c) → Result)
  (cost_pos: ∀ (c₀ c₁: Cost), c₀ ≤ c₀ + c₁)
  (has_cost_mono: ∀ p c₀ c₁, has_cost p c₀ → c₀ ≤ c₁ → has_cost p c₁)
```

The `model` structure in the provided code defines a complexity model, which consists of several components that enable us to reason about the complexity of programs and their behavior on specific inputs.  This is a “blueprint” for a computational model, meaning that it is an abstract representation of one. Since there are many computational models, this defines the generic characteristics of it. 

The `Program`, `Data`, `Result`, and `Cost` types represent the programs, input data, output results, and cost measures, respectively. These generic types allow the model to be specialized for various scenarios. The `data_equiv` and `result_equiv` components define equivalence relations on the `Data` and `Result` types, allowing us to compare and reason about input data and output results in the context of the complexity model.

The `cost_lt` and `cost_add` components provide a preorder and an addition operation on the `Cost` type, respectively, enabling meaningful comparison and manipulation of costs. 

The `apply` function takes a program and input data, returning a new program that represents the result of applying the original program to the input data. A Turing machine could be defined like this, where the `Data` input type would the state of the tape and the `Program` would be other parts of a Turing machine (its head position, the current state, etc…).  But some computational model will just ignore the `Data` input and just “step” the program forward. A cellular automaton could be thought of like this because it does not have an input, it just steps the program forward based on its current state. 

The `has_cost` predicate associates a program with a cost, asserting that the given program has the specified cost in the complexity model. The `result` function takes a program and a proof of its cost, returning the output result associated with the program and the given cost. The `cost_pos` and `has_cost_mono` axioms express properties of the cost function, such as its non-negativity and monotonicity.

These components together form a comprehensive framework for reasoning about the complexity of programs and their behavior on different inputs. By defining the appropriate instances for the model, we can apply this framework to analyze the complexity of specific algorithms, such as merge sort, and gain insights into their performance characteristics.

### Hierarchical Memory Model

Now we can talk about a certain computational model that the Lean code in this GitHub repository called the hierarchical memory model (hmem). The hmem is a powerful and expressive framework for modeling the complexity of programs and their execution in a hierarchical memory setting. It is built on several key components, such as programs, instructions, memory, thunks, and traces, which work together to represent and reason about computation and its associated costs.

In the hmem, programs consist of sequences of instructions that can manipulate memory and make calls to other instructions or perform recursion. The available instructions include load, store, and branch operations, as well as comparisons and arithmetic operations. These instructions enable the representation of a wide variety of algorithms, including merge sort.

The memory model in the hmem framework is hierarchical, representing different levels of memory in typical computers, such as registers, cache, and main memory. Memory is modeled as a mapping from memory locations to values, allowing the storage and retrieval of data during computation. This hierarchical memory model plays a crucial role in the analysis of programs' complexity, as it considers the cost of accessing different levels of memory, which is an essential aspect of real-world computation.

Thunks represent suspended computations and play a crucial role in the model, allowing for laziness and delayed evaluation of expressions. Traces capture the history of a program's execution, including the branching decisions, steps taken, and memories encountered during the process. This information is used to analyze the program's complexity, such as the number of memory accesses, branches, and recursive calls made during execution.

Using the hmem, the merge sort algorithm is represented as a program, and its complexity is analyzed in terms of the number of memory accesses, branches, and recursive calls made during execution. The proof of merge sort's time complexity is accomplished by relating the program to a specific runtime environment, which takes into account the program's memory and time cost. 

This is what the hmem computation model is defined as in Lean. It is an instance of the computational model defined above, so it must provide values for each required type that composes a computational model. 

```
def hierarchical_model (μ: Type) [decidable_eq μ] [has_zero μ] [has_one μ] [ne_zero (1:μ)]: complexity.model :=
⟨ (program μ × list (memory μ)),
  memory μ,
  memory μ,
  ℕ,
  ⟨ (=), eq.refl, @eq.symm _, @eq.trans _ ⟩,
  ⟨ (=), eq.refl, @eq.symm _, @eq.trans _ ⟩,
  infer_instance,
  infer_instance,
  λ p_a a, (p_a.fst, a::p_a.snd),
  λ p_a, (prod.fst p_a).has_time_cost (build_arg p_a.snd),
  λ p_a _ h, program.result p_a.fst (build_arg p_a.snd) ⟨_, h⟩,
  λ _ _, le_self_add,
  λ _ _ _ hrc₀ hc, program.time_cost_mono hrc₀ hc ⟩
```

The model introduces a variable `μ` which represents the type of data it can store. The constraints on this data type include having a zero type and a one type. This is so some basic functions can be applied to the data. 

The `Program` type consists of a tuple containing a `program` of type and a list of memory states. The `program` is a different type that is defined to be a list of instructions. The `memory` is a recursive tree type that has leaves of type `μ`. Combining these both together make the entire `Program` type. 

A `program` also has methods on it like  `has_time_cost` which are defined else where but define a proof of the fact that a certain program has a certain cost function. 

Both input data and output results are represented by memory states, and the cost is given by a natural number. The `apply` function is used to update the program and memory states when the program is applied to input data.  The model also includes axioms for cost properties, such as non-negativity and monotonicity, which are essential for reasoning about the complexity of algorithms within this model.

We can now dive deeper into how the instructions work. The type definition of the instructions follows. 
```
inductive memory_operation
| COPY
| MOVE
| SWAP

inductive instruction (α: Type u)
| vop {n: ℕ} (op: vector α n → α) (dst: source α) (src: vector (source α) n): instruction
| mop (op: hmem.instruction.memory_operation) (dst src: source α): instruction
| clear (dst: source α): instruction
| ite {n: ℕ} (cond: vector α n → Prop) [Π {v}, decidable (cond v)] (src: vector (source α) n) (branch: list instruction): instruction
| call (func: list instruction) (dst src: source α): instruction
| recurse (dst src: source α): instruction
```

There are 6 instructions in the hmem. 
- `vop` is a vector operation. This allows you to apply some operation to a vector in memory
- `mop` is a memory operation. There are three kinds of operations that manipulate data in various ways. Its parameters are a source and destination. 
- `clear` removes an element from memory. 
- `ite` is an “if then else” operation. It takes a condition and two possible branches to go down and executives the respective one depending on the condition. 
- `call` allows a program to call another instruction
- `recurse` allows a program to call itself.

Each of these instructions can be combined to make a full program. There are many theorems in the repository that prove that these instructions combined in certain ways have to take a certain amount of cost.  Now we show the formalization for the merge sort program.
```
def merge_sort {μ: Type*} [decidable_eq μ] [has_zero μ] [has_one μ] [ne_zero (1:μ)] (cmp: program μ): program μ :=
[ -- l
  instruction.ifz (source.imm 1 source.nil) [
    -- list.nil ∨ [a]
  ],
  -- 1 [1 a [1 b l]]
  instruction.call (split μ) source.nil source.nil,
  -- 1 l₀ l₁
  instruction.recurse (source.imm 0 source.nil) (source.imm 0 source.nil),
  -- 1 (ms l₀) l₁
  instruction.recurse (source.imm 1 source.nil) (source.imm 1 source.nil),
  -- 1 (ms l₀) (ms l₁),
  instruction.call (merge cmp) source.nil source.nil
  -- ms l
]
```

The $\mu$ variable is the same that is the input to the hmem. The `cmp` variable is a comparison function that merge sort needs to know how to compare the data types it is sorting. This whole structure is defined as a `program`, which we have already said is a list of instructions. The instructions define a recursive function that splits an input array, recursively sorts the sub arrays, then merges the resulting sub arrays. The first instruction also represents the base case of if an empty array or and array with a single element is passed in.

This split and merge programs are defined elsewhere are more complicated but use the same structure of having an array of instructions that performs the functions. This mechanism is useful because each instruction has a definition cost of running, and we can prove aspects of this function's behavior because of it, this is the basis of how we prove the complexity of these functions. 

### Formalizing complexity

To begin to show the complexity of an algorithm, we must first prove that a certain algorithm has a certain trace. Here is the definition for merge sorts trace.
```
def merge_sort_trace {α: Type*} [has_encoding α μ] (fcmp: α → α → Prop) [dcmp: decidable_rel fcmp] (pcmp: program μ):
  list α → trace μ
| [] := ⟨encode (@list.nil α), [tt], 1, [], []⟩
| [a] := ⟨encode [a], [tt], 1, [], []⟩
| l := match list.split l with
  | (as, bs) :=
    ⟨encode (list.merge_sort fcmp l), [ff], 5,
     [encode l,
      encode (list.merge_sort fcmp as, list.merge_sort fcmp bs)],
     [encode as, encode bs]⟩
  end
```

The function constructs a trace for the merge sort algorithm applied to a list of elements of type `α`, given a comparison function `fcmp` and a program `pcmp` that implements the comparison. The trace represents the sequence of memory states and decisions made by the merge sort algorithm during its execution, providing a detailed account of the algorithm's behavior and its interaction with memory.

A trace is crucial in formalizing the complexity of merge sort, as it allows us to capture not only the final result but also the intermediate steps and memory states throughout the execution. By employing the `encode` function, the elements of type `α` are transformed into memory states, which can then be processed by the hmem. This encoding ensures that the complexity analysis is conducted within the hierarchical memory model, taking into account the actual memory states and operations performed by the merge sort algorithm.

There are then theorems that prove that the merge sort program defined above has to have the defined trace. This means we can work with the trace of the program to prove its complexity instead of the program itself. 

Now we can show the definition for complexity. The following definition is an alias for a proof. Whenever the word `complexity` is seen, this proof can be used in its place. 

```
def complexity {α: Type*} {β: Type*} (m: complexity.model) (f: α) (c: β): Prop :=
  ∃ (program: m.Program), complexity.witness m program f c
```

We are defining `complexity` to be a proposition that some function `f` can be computed in some computational model `m` with some cost `c`. This proposition is defined to be true when there is some program in the model that can “witness” the program costing that much. A witness is a mechanism that surrounds a computation model and watches each `apply` call that happens (remember that apply means a step of the program) and ensures that each program that is outputted is valid. This prevents programs from cheating and returning costs that don't match their program execution. 

The main result of this repository is the proof of Merge Sort's time complexity. A simplified version of the theorem is shown below, the proof has been omitted. 

```
def merge_sort_cost (n: ℕ) : ℕ := 22 * n * (nat.clog 2 n) + 2

theorem merge_sort_complexity:
  complexity 
	  (hmem.encoding.runtime_model μ) 
	  (list.merge_sort cmp) 
	  merge_sort_cost
```

`μ` and `cmp` are defined the same as they have been. We first introduce the `merge_sort_cost` function, which is a function that maps the length of the list to the exact complexity of the algorithm. Then we use the above definition of `complexity` to define the theorem. The proof consists of a bunch of algebra counting the exact number of instructions the algorithm makes on any input.

That concludes the overview of the GitHub repo that defines computational model and basic complexity. While this is still very much in progress, I think the path that is being going down is promising and is a good approach to show the exact time complexity of programs. The repo has many more lemmas and proofs for them that are the building block for how the hmem works and for keeping track of the number of instructions that a program makes. 

## My contributions to repository

Now I want to move to the part that I've contributed that adds more time complexity structures to Lean. I'll define a couple structures and theorems that show that merge sort is in the complexity class P. I first showed the asymptotic time complexity of merge sort is `nlgn` by using the precise result. Then I'll define the complexity class P and then prove a theorem that shows that Merge Sort is in the complexity class P. 

All of the code I contributed can be found in the src directory included in the zip file. The file I worked on was `complexity/class.lean` 

### Algorithm type

First, I define a new structure `Algorithm`

```
structure Algorithm :=
  (Name : string)
  (Model : complexity.model)
  (Input: Type*)
  (Output: Type*)
  (Function: Input → Output)
  (Cost: ℕ → ℕ)
  (Asymptotic_Cost: ℕ → ℕ)
  (complexity: complexity Model Function Cost)
  (asymptotic_complexity: is_O at_top Cost Asymptotic_Cost)
```

An algorithm is a collection of types that defines the input and output to a function. It also requires a cost function and a complexity proof showing that the algorithm in fact has a complexity in a certain computational model. It also requires an asymptotic cost and a function proving that it is big O of the precise cost. This structure is useful to discuss collections of properties of algorithms and define relationships between them. Ultimately, I want to prove results about a certain `Algorithm` being in P.

Note that the `Algorithm` structure needs a model of computation. This is a slightly strange way to think of algorithms, which we normally assume can be performed in any computational model. But since the computational model defined in this repository is so generic, it isn't obvious that an algorithm in one complexity class would be in the same complexity class in another model. So I decided to stick with this definition. If a Turing machine computation model was formalized, then one could either reprove the complexity of an algorithm in that model, or devise a way of translating from one model to the other and add the cost of the translation to the cost function for that algorithm.

Here is the merge sort algorithm defined in this structure
```
def merge_sort
  (μ : Type )
  [decidable_eq μ] [has_zero μ] [has_one μ] [ne_zero (1:μ)]
  (cmp: ℕ → ℕ → Prop)
  [decidable_rel cmp]
  : Algorithm :=
  {
    Name := "merge_sort",
    Model := hmem.encoding.hierarchical_model μ,
    Input := list ℕ,
    Output := list ℕ,
    Function := list.merge_sort cmp,
    Cost := merge_sort_cost,
    Asymptotic_Cost := merge_sort_asymptotic_cost,
    complexity := merge_sort_complexity,
    asymptotic_complexity := merge_sort_asymtotic_complexity
  }
```

The merge sort algorithm takes in a list of natural numbers and outputs another list of natural number. The cost and proof of the cost are the types defined previously. The asymptotic cost and proof of that cost will be defined in the next section.

### Asymptotics 


```
def merge_sort_asymptotic_cost (n: ℕ) : ℕ := n * (nat.clog 2 n)
```

The definition of the merge sort asymptotic cost just says that merge sort take `nlg(n)` time. Ideally, this would be defined in a way such that there are no leading constant factors or lesser terms being added to the cost function. I search for a way to do this in Lean and couldn't find an immediate solution. My best idea of how to do this is to enforce that a proof is provided that shows that this function is Big O and Big $\Omega$ of itself. This could be a first step in ensuring there aren't constants in the cost function. But really this isn't required for proofs, it is just nice to have to keep people who use this from defining incorrect upper bounds. 

Here is the theorem that proves the upper bound relation. 

```
theorem merge_sort_asymtotic_complexity:
  is_O at_top merge_sort_cost merge_sort_asymptotic_cost
```

We use the `is_O` and `at_top` types that are already defined in Lean's standard math library. The `is_O` requires us to provide a proof that there is a constant $c$ that you can multiply the asymptotic cost by such there is some value of `n` where the asymptotic cost is always greater than the provided cost. The `at_top` is a limit that defines how `is_O` works, it says that asymptotic function must overtake the provided function as they approach $\infty$. There is also a `at_bottom` that enforced the function overtakes as they approach -$\infty$. 

The proof of this theorem relies on the lemma 
```
lemma n_log_n_O
  (a: ℕ)
  (b: ℕ)
  : is_O at_top (λ n, a * n * nat.clog 2 n + b) (λ n, n * nat.clog 2 n) :=
```

This lemma contains the actual algebra that prove that there is a point that some large value that can be multiplied to `nlgn` such that it can take over any other function of the form `a*nlgn + b`. In this proof, I use the value `2 * a
to multiply the asymptotic cost by and then show that it overtakes the function at $n=\frac{2b}{a}$. 

Lean has a concept of tactics that you can use to auto prove certain theorems of a particular value. For example, the `cc` tactic can prove almost any logical statement of ands, ors, and nots. I believe that there can be some clever tactics created in Lean that can automatically prove lemmas like the above one using some simple principles. That would make proving complexity a much simpler task, instead of having to do a large amount of algebra for each complexity result. 

### P Complexity class

My next goal was to define the P complexity class. P is normally defined as follows.

$$
 P = \bigcup_{c = 1}^{\infty}  \{ L \mid \exists \text{ a TM that runs in } O(t(n^c)) \text{ and decides L} \}
 $$

Since the definitions that have been defined previously don't include Turing machines or languages, a different approach is needed to define P. I used the definition of an algorithm and its cost to make a rather simple definition of if an algorithm is in the complexity class P.

```
def in_P (A : Algorithm) : Prop :=
  ∃ (n : ℕ), ∃ (c: ℕ), is_O (A.Cost) (λ n, n^c)
```

And the actual definition of the set is as follows
```
def complexity_class_p : set Algorithm :=
  { A : Algorithm | in_P A }
```

Informally P is this is the set of algorithms whose cost function can be bounded by a polynomial of form $n^c$.

This is equivalent to the above definition. The cost function of an algorithm is the same as a TM running in $O(COST)$. Merge sort is not a decision problem a function that is applied to a list of element. And my definition of P isn't in term of decision problems, but rather in terms of functions that map the input type to the output type. I believe this to be a more general definition that can be applied to a wide range of computational models, but more work needs to be done to show that you could formulate Turing machine in this way. 

Then to show that an algorithm is a member of the P complexity class, you just need to provide a proof about the bounds of its cost function. I'll now show a theorem that merge sort is in the complexity class P and provide the proof. I've commented each line to explain what it is doing. The proof relies on a lemma of the form.

```
lemma pow_big_o_n_log_n
  {n: ℕ}
  (a: ℕ)
  (b: ℕ)
  (c: ℕ)
  (h: c > 2)
  : is_O at_top (λ n, a * n * nat.clog 2 n + b) (λ n, n ^ c)
```

This lemma just states that once $n^c$ is an upper bound for `nlgn` when `c > 2`. Which is easy to see, $n^2$ always takes over $nlgn$ after some large value of $n$. 

Now here is the theorem and the full proof

```
-- Merge Sort is in P
theorem merge_sort_in_P : in_P (merge_sort μ cmp) :=
begin
  -- Replace the in_P label with its defintion
  rw in_P,

  -- Replace merge sort with its defintion
  unfold merge_sort,

  -- Replace cost_in_P with its definition
  simp only [cost_in_P, Algorithm.Cost],

  -- Prove that merge_sort_cost is in O(n^3)
  have h: is_O at_top merge_sort_cost (λ n, n ^ 3),
  {
	-- Apply the lemma
    apply pow_big_o_n_log_n,
    -- Prove that 3 > 2
    exact dec_trivial,
  },

  -- Pick a constant c as 3
  use 3,

  -- The goal is now just h
  exact h,
end
```

The basic idea of the proof is to pick a value for c in $n^c$ and such that it takes over `nlgn` at some point. Many simplification statements are used to convert from the language of asymptotics to an algebraic equation. There are a lot of places I could improve the proof by learning more tactics in Lean and by utilizing the asymptotics library more, but this proof as is works. 


## Conclusion and future work

In the end, I wasn't able to prove every theorem and lemma and got nowhere near the best proofs for the ones I did prove. As I learned more about Lean, I kept reworking proofs with new ideas, and some types are not structured correctly or consistently because of that. I wasn't able to fully comprehend the complexity formalizations in the GitHub repository because of the sheer amount of lines of code and theorems, but after talking to the author of the repository and a lot of time struggling with it, I understood the basics well enough to write a report on how it works. 
I'm glad I spent the time learning the basics of Lean. I think computer assisted proofs are going to be heavily utilized in the future and can speed up the work of mathematicians and computer scientists by large factors. Just more work needs to be put into formalizing the proofs we have already created. 

If I were to continue working on this project, I would want to aim for a formalization of the complexity class NP. I would probably use the certificate definition, since the nondeterministic Turing machine assumes you are working in a Turing machine computational model. The general approach I'd take is to define the class to be algorithms that are in P when there exists some polynomial length input you can provide them to help them.

I'd also like to try to come up with some tactics that automate some tedious tasks of the proofs that I had to do. Lots of the asymptotics seems like they would be straight forward to automate. But also automatically determining the complexity class of an algorithm based on its cost function would be beneficial. 

Another case I'd like to think though is how cost functions are proved. Currently, you must first prove the precise cost function of an algorithm, then you can prove the bounds of that cost function. This can certainly be tedious for some algorithms, so having a way to prove the asymptotic cost from the beginning and not have to prove the precise cost would allow for more rapid progress on formalizing algorithms. I think there are sensible approaches to this, like defining a different type of complexity proof that needs an asymptotic witness instead of a precise witness.

I also think documentation is important. This repository had very little documentation and that was apart of my motivation for doing a write up for it. I plan on creating a readme for the repository that adapts this reports into an overview of the codebase. 

I'd also like to talk to others that are working on formalizing complexity in Lean and try to come up with an approach that suits everyone's needs, so the code can be added to the standard math library. Once code in there, anyone who is using Lean can search for a proof, and if there were complexity results in the standard library I think it would be a great advantage to be able to quickly look up with some result has already been proven. 


## References

Andrew Carter, Lean Complexity Theory, (2022), GitHub repository, 
https://github.com/calcu16/lean_complexity


