import tactic

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

def alphabet := string

def tile_type := U2 -> (alphabet × nat)

def tile_1 : tile_type :=
  λ direction,
    match direction with
    | U2.north := ("a", 2)
    | U2.east  := ("a", 1)
    | U2.south := ("c", 1)
    | U2.west  := ("d", 1)
    end

def tile_2 : tile_type :=
  λ direction,
    match direction with
    | U2.north := ("a", 2)
    | U2.east  := ("a", 1)
    | U2.south := ("a", 1)
    | U2.west  := ("c", 1)
    end

def T := tile_type

def assembly_dom := {p : Z2 // p = (0,0) ∨ p = (0,1)}
def assembly_range := {t : tile_type // t = tile_1 ∨ t = tile_2}

def assembly_map : Π (p : assembly_dom), assembly_range :=
  λ (p : assembly_dom),
    match p with
    | ⟨(0, 0), _⟩ := ⟨tile_1, or.inl rfl⟩
    | ⟨(0, 1), _⟩ := ⟨tile_2, or.inr rfl⟩
    | _ := ⟨tile_1, or.inl rfl⟩ -- default case
    end

def assembly_dom_2 := {p : Z2 // (0 ≤ p.fst ∧ p.fst ≤ 10) ∧ (0 ≤ p.snd ∧ p.snd ≤ 10)}

def assembly_map_2 : Π (p : assembly_dom_2), assembly_range :=
  λ (p : assembly_dom_2),
    if p.val.snd % 2 = 0 then ⟨tile_1, or.inl rfl⟩
    else ⟨tile_2, or.inr rfl⟩


-- 1xn assembly
def assembly_1xn_dom (n: ℕ) := {p : Z2 // p = (0,0) ∨ p = (0, n)}


