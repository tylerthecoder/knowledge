import tactic




def A := {n : ℕ // n > 0} -- This defines A as a subtype of ℕ, where each element is > 0

def B := {n : ℕ  // n ≠ 0} -- This defines B as a subtype of ℤ, where each element is not equal to 0

def map_A_to_B (a : A) : B := ⟨
  a.1 + 1,
    begin
    have h : a.1 > 0 := a.2,
    linarith,
  end
⟩
