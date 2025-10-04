-- 命題論理のトートロジー体系
namespace PropositionalLogic

-- 命題論理の論理式
inductive Formula where
  | var : String → Formula        -- 要素命題記号
  -- 以下、結合子
  | top : Formula                 -- 真 (⊤)
  | bot : Formula                 -- 偽 (⊥)
  | neg : Formula → Formula       -- 否定 (¬)
  | conj : Formula → Formula → Formula  -- 連言 (∧)
  | disj : Formula → Formula → Formula  -- 選言 (∨)
  | impl : Formula → Formula → Formula  -- 含意 (→)
  | bicon : Formula → Formula → Formula -- 同値 (↔)
  deriving Repr, DecidableEq

-- 評価（真理値割り当て）
def Valuation := String → Bool

-- 論理式の評価を帰納的に定義
def eval (v : Valuation) : Formula → Bool
  | Formula.var p => v p
  | Formula.top => true
  | Formula.bot => false
  | Formula.neg φ => !eval v φ
  | Formula.conj φ ψ => eval v φ && eval v ψ
  | Formula.disj φ ψ => eval v φ || eval v ψ
  | Formula.impl φ ψ => !eval v φ || eval v ψ
  | Formula.bicon φ ψ => eval v φ == eval v ψ

-- トートロジーの定義：すべての真理値割り当てで真となる論理式
-- 恒真式とは、要素命題記号にどのように真理値を与えても常に1となるような論理式
def isTautology (φ : Formula) : Prop :=
  ∀ v : Valuation, eval v φ = true

-- 充足可能性の定義
def isSatisfiable (φ : Formula) : Prop :=
  ∃ v : Valuation, eval v φ = true

-- 矛盾（充足不可能）の定義
-- 矛盾式とは、要素命題記号にどのように真理値を与えても常に真理値0になるような論理式
def isContradiction (φ : Formula) : Prop :=
  ∀ v : Valuation, eval v φ = false

-- 論理的帰結の定義
def entails (Γ : List Formula) (φ : Formula) : Prop :=
  ∀ v : Valuation, (∀ ψ ∈ Γ, eval v ψ = true) → eval v φ = true

-- 基本的なトートロジーの例（定義のみ）

-- 排中律: P ∨ ¬P
def exampleExcludedMiddle (p : String) : Formula :=
  Formula.disj (Formula.var p) (Formula.neg (Formula.var p))

-- 矛盾律: ¬(P ∧ ¬P)
def exampleNonContradiction (p : String) : Formula :=
  Formula.neg (Formula.conj (Formula.var p) (Formula.neg (Formula.var p)))

-- 同一律: P → P
def exampleIdentity (p : String) : Formula :=
  Formula.impl (Formula.var p) (Formula.var p)

-- モーダスポネンス: (P ∧ (P → Q)) → Q
def exampleModusPonens (p q : String) : Formula :=
  Formula.impl
    (Formula.conj (Formula.var p) (Formula.impl (Formula.var p) (Formula.var q)))
    (Formula.var q)

-- ド・モルガンの法則: ¬(P ∧ Q) ↔ (¬P ∨ ¬Q)
def exampleDeMorgan1 (p q : String) : Formula :=
  Formula.bicon
    (Formula.neg (Formula.conj (Formula.var p) (Formula.var q)))
    (Formula.disj (Formula.neg (Formula.var p)) (Formula.neg (Formula.var q)))

-- ド・モルガンの法則: ¬(P ∨ Q) ↔ (¬P ∧ ¬Q)
def exampleDeMorgan2 (p q : String) : Formula :=
  Formula.bicon
    (Formula.neg (Formula.disj (Formula.var p) (Formula.var q)))
    (Formula.conj (Formula.neg (Formula.var p)) (Formula.neg (Formula.var q)))

-- トートロジーに関する基本定理

-- φがトートロジーであることと、¬φが充足不可能であることは同値
theorem tautology_iff_not_satisfiable_neg (φ : Formula) :
  isTautology φ ↔ ¬isSatisfiable (Formula.neg φ) := by
  constructor
  · intro h ⟨v, hv⟩
    have := h v
    unfold eval at hv
    rw [this] at hv
    contradiction
  · intro h v
    by_cases heval : eval v φ = true
    · exact heval
    · exfalso
      apply h
      exists v
      unfold eval
      cases hφ : eval v φ
      · rfl
      · contradiction

-- φが矛盾であることと、¬φがトートロジーであることは同値
theorem contradiction_iff_tautology_neg (φ : Formula) :
  isContradiction φ ↔ isTautology (Formula.neg φ) := by
  constructor
  · intro h v
    unfold eval
    have := h v
    rw [this]
    rfl
  · intro h v
    have := h v
    unfold eval at this
    cases hφ : eval v φ
    · rfl
    · rw [hφ] at this
      contradiction

-- 簡単な使用例
#eval eval (fun _ => true) (exampleExcludedMiddle "P")    -- true
#eval eval (fun _ => false) (exampleExcludedMiddle "P")   -- true
#eval eval (fun _ => true) (exampleIdentity "P")          -- true

-- 論理式から変数名のリストを取得
def getVars : Formula → List String
  | Formula.var p => [p]
  | Formula.top => []
  | Formula.bot => []
  | Formula.neg φ => getVars φ
  | Formula.conj φ ψ => (getVars φ ++ getVars ψ).eraseDups
  | Formula.disj φ ψ => (getVars φ ++ getVars ψ).eraseDups
  | Formula.impl φ ψ => (getVars φ ++ getVars ψ).eraseDups
  | Formula.bicon φ ψ => (getVars φ ++ getVars ψ).eraseDups

-- すべての可能な真理値割り当てを生成
def allValuations (vars : List String) : List Valuation :=
  let rec go (vs : List String) : List (List (String × Bool)) :=
    match vs with
    | [] => [[]]
    | v :: rest =>
      let subVals := go rest
      subVals.map (fun vals => (v, true) :: vals) ++
      subVals.map (fun vals => (v, false) :: vals)
  (go vars).map fun assignments =>
    fun var => (assignments.find? (fun p => p.1 == var)).map (·.2) |>.getD false

-- トートロジーかどうかをチェック（計算可能版）
def checkTautology (φ : Formula) : Bool :=
  let vars := getVars φ
  let valuations := allValuations vars
  valuations.all (fun v => eval v φ)

-- トートロジーチェック結果を文字列に変換
def isTautologyString (φ : Formula) : String :=
  if checkTautology φ then "トートロジー (恒真)" else "トートロジーではない"

-- 充足可能性チェック（計算可能版）
def checkSatisfiable (φ : Formula) : Bool :=
  let vars := getVars φ
  let valuations := allValuations vars
  valuations.any (fun v => eval v φ)

-- 矛盾チェック（計算可能版）
def checkContradiction (φ : Formula) : Bool :=
  let vars := getVars φ
  let valuations := allValuations vars
  valuations.all (fun v => !eval v φ)

-- 真理値表を文字列として生成
def truthTable (φ : Formula) : String :=
  let vars := getVars φ
  if vars.isEmpty then
    s!"変数なし: {eval (fun _ => false) φ}\n"
  else
    let valuations := allValuations vars
    let header := vars.foldl (fun acc v => acc ++ v ++ " | ") "" ++ "結果\n"
    let separator := String.mk (List.replicate (header.length) '-') ++ "\n"
    let rows := valuations.map fun v =>
      let values := vars.map (fun var => if v var then "T" else "F")
      let result := if eval v φ then "T" else "F"
      values.foldl (fun acc val => acc ++ val ++ " | ") "" ++ result ++ "\n"
    header ++ separator ++ rows.foldl (· ++ ·) ""

end PropositionalLogic
