

I would like to give a write and analyse of this github repository https://github.com/prakol16/circuits or https://github.com/calcu16/lean_complexity. Thye both attempt to formalize a compleixyt class and defines polynomial time. Some of this code should be incorporated into the lean mathlib at some point. I would like to prove that some algorithm is in the comleixty class P. I would like to choose that algorithm after I understnad the code more so I know what is feasible but I think it will pertain to performing some operation on an array. I would also like to wait to choose which code base I use until I understand a bit more to see which is more applicaple.





Different implementations
https://github.com/leanprover-community/mathlib/pull/14494


https://github.com/calcu16/lean_complexity

https://github.com/prakol16/circuits


Looks like most complexity algorithms assert that you can sort an array in 8 * n log n the length of the array.

Might be a bit different to actually define the complexity algorithms.




## Notes

Defines a generic model of computation.

Hmem is a type of model of computation. It is similar to how memory is accessed on a normal computer. There are different levels of memory. On average memory at location x is accessed in O(lg x) time.

I could write a paper on the current models of computation then attempt to formalize the abstract tile assembly and write simple proofs about it.

Could also try to find a way of defining the P complexity class and that the merge sort algorithm is a part of it. 


General overview of his stuff. 
Computational model has a program, data, and a cost function

For hmem, you define a trace which is a histroy of the program execution. you can porve that some program will always output that trace, and that the trace is of a certain size. 


**Project Description**
Your submission should include:

(1) I neatly written report as a formatted PDF with a clear description of your project including: a description of the problem, your approach/design, and the results. Different projects may require additional information.

(2) If relevant, any code and/or data files needed to reproduce your results.

(3) Other supplementary information related to your project (depending upon the type of project).

Include all files in a single zip folder.


Outline of profile report

Intro
- Motivation: I like formalizing things and I think that formalizaing complexity classes so we can programatically check correctness is important. 
- Project Description: Explain the formalization made in a github repo for computational models and the complexity of merge sort. My stretch goal is move to asymtotics and formally define the complexity class P. 
What is lean
- What is lean theorem prover? 
How does this formalization work? 
- Uses a formal definition of a computation model 
- explain formalization
- Uses hierachical memory model to formalizae the complexity of merge sort. 
How doe we add asymtotics? 
How to define the P complexity class. 
Results
- Did I finish everything? 
