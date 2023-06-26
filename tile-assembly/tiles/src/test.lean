-- A vector that represents a direction in the tile (0,1) = north, (1,0) = east, (0,-1) = south, (-1,0) = west

-- Discrete Euclidean plane Z^2
def Z2 := ℤ × ℤ

inductive U2
| north : U2
| east : U2
| south : U2
| west : U2

def unit_2_to_int_pair (t : U2) : Z2 :=
  match t with
  | U2.north := (0, 1)
  | U2.east  := (1, 0)
  | U2.south := (0, -1)
  | U2.west  := (-1, 0)
  end

-- Set of all 2-element subsets of a set X
def two_element_subsets {α : Type} (X : set α) : set (set α) :=
  {Y | ∃ (x y : α), x ∈ X ∧ y ∈ X ∧ x ≠ y ∧ Y = {x, y}}

-- Undirected graph
structure graph :=
  (V : set Z2)
  (E : set (set Z2))
  (edges_in_two_element_subsets : ∀ e ∈ E, e ∈ two_element_subsets V)

-- Binding function on a graph
def binding_function (G : graph) := G.E → ℕ

-- Binding strength of a binding function on a graph
def binding_strength (β : binding_function) (G : graph) : ℕ :=
  sorry -- This requires a more complex definition involving sets and functions

-- Binding graph
structure binding_graph :=
  (G : graph)
  (β : binding_function G)
  (τ : ℕ)
  (τ_stable : binding_strength β G ≥ τ)


-- Partial function from a set X to a set Y
def partial_function {α β : Type} (X : set α) (Y : set β) :=
  {f : α → β // ∃ (D : set α), D ⊆ X ∧ f = set.restrict f D}


def alphabet := string

def tile_type := U2 -> (alphabet × nat)

def example_tile : tile_type :=
  λ direction,
    match direction with
    | U2.north := ("a", 2)
    | U2.east  := ("a", 1)
    | U2.south := ("c", 1)
    | U2.west  := ("d", 1)
    end

def T := tile_type

def TConfiguration := ℤ × ℤ -> T


