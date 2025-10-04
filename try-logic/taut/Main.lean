import Taut

open PropositionalLogic

def main : IO Unit := do
  IO.println "命題論理のトートロジー体系"
  IO.println ""

  -- 排中律の例
  let p := "P"
  let formula := exampleExcludedMiddle p
  IO.println s!"排中律 (P ∨ ¬P): {repr formula}"
  IO.println s!"全てtrueの場合の評価: {eval (fun _ => true) formula}"
  IO.println s!"全てfalseの場合の評価: {eval (fun _ => false) formula}"
  IO.println ""

  -- 同一律の例
  let formula2 := exampleIdentity p
  IO.println s!"同一律 (P → P): {repr formula2}"
  IO.println s!"全てtrueの場合の評価: {eval (fun _ => true) formula2}"
  IO.println s!"全てfalseの場合の評価: {eval (fun _ => false) formula2}"

  -- p11 9) 連言の吸収律
  -- A and (A or B) equivalence A
  let A := Formula.var "A"
  let B := Formula.var "B"
  let absorptiveLaw := Formula.bicon (Formula.conj A (Formula.disj A B)) A
  IO.println ""
  IO.println s!"連言の吸収律: A ∧ (A ∨ B) ⇔ A"
  IO.println s!"結果: {isTautologyString absorptiveLaw}"
  IO.println s!"真理値表:"
  IO.println (truthTable absorptiveLaw)
