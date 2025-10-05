/-- MP: modus ponens -/
example (P Q : Prop) (h : P → Q) (hp : P) : Q := by
  apply h -- ⊢ Q → ⊢ P に変わる
  exact hp -- 仮定 P を使う

example(P Q : Prop): (¬P ∨ Q) → (P → Q) := by
  intro h -- 仮定 ¬P ∨ Q を h とする
  cases h with
  | inl hp => -- hp : ¬P
    intro p -- 仮定 P を p とする。この時点で ¬P と P が仮定にあるので矛盾している
    exfalso -- 矛盾から何でも言える
    exact hp p -- hp : ¬P, p : P なので矛盾
  | inr hq => -- hq : Q
    intro p -- 仮定 P を p とする
    exact hq -- hq : Q を使えば良い

example (P Q : Prop) : ¬(P ∨ Q) ↔¬P∧¬Q := by
  constructor
  · intro h -- h : ¬(P ∨ Q) (→を証明)
    constructor -- 論理積を2つに分解
    · intro hp
      exfalso
      apply h
      left
      exact hp
    · intro hq
      exfalso
      apply h
      right
      exact hq
  · intro h -- h : ¬P ∧ ¬Q (←を証明)
    cases h with
    | intro hp hq =>
      intro h' -- h' : P ∨ Q
      cases h' with
      | inl hp' =>
        apply hp
        exact hp'
      | inr hq' =>
        apply hq
        exact hq'
