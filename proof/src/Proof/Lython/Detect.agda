{-# OPTIONS --safe #-}

-- Deciding the invalidities, and what a step can and cannot introduce.
--
-- A predicate nobody can evaluate is a specification, not a check. These are
-- the procedures a pass would actually run, with soundness proofs -- which is
-- the part that stops "we added a checker" from being the whole claim.

open import Proof.Memory.Element using (ElemSig)

module Proof.Lython.Detect (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; if_then_else_; _∧_; _∨_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_; map)
open import Data.Maybe using (Maybe; just; nothing; maybe′)
open import Data.Nat using (ℕ; zero; suc; _≟_; _<_; s≤s; z≤n)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong; subst; subst₂)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; Life; live; finalizing; dead;
  RuntimeCount; counted; immortal; sameObj; sameObj-refl; sameObj-sound;
  ≡ᵇ-refl; ≡ᵇ-sound; ≡ᵇ-sym)
open import Proof.RC.OwnerSite using (OwnerSite; SiteMap; ThreadId; sameSite;
  sameSite-false; sameSite-refl; sameSite-sound; sameSite-sym;
  Holds; holds-here; holds-there; holds-positive; logicalRC)
open import Proof.RC.Machine Sig
open import Proof.Program.Syntax using (Var; Instr; alloc; init; move; dup; drop; borrow)
open import Proof.Program.Env
open import Proof.Program.Step Sig
open import Proof.Lython.Invalid Sig

------------------------------------------------------------------------
-- Env facts the checker needs.

lookup-bind-here : ∀ (es : Env) (x : Var) (b : Binding) →
                   lookupVar (bindVar es x b) x ≡ just b
lookup-bind-here es x b rewrite sameVar-refl x = refl

-- Distinctness is needed because binding SHADOWS: `borrow x x` would rebind the
-- anchor to the borrow itself, and then the anchor is trivially present. That
-- case is not a dangling borrow but it is not a sensible program either, and
-- the hypothesis is where the model says so out loud.
lookup-bind-there : ∀ (es : Env) (x y : Var) (b : Binding) →
                    sameVar x y ≡ false →
                    lookupVar (bindVar es x b) y ≡ lookupVar es y
lookup-bind-there es x y b ne rewrite ne = refl

------------------------------------------------------------------------
-- The dangling-borrow check.
--
-- Computable, and it needs no analysis: the anchor is recorded in the binding,
-- so the whole question is a second lookup. That is the payoff of putting the
-- anchor in `Mode.borrowed` rather than leaving borrows anonymous.

danglingAnchor : Env → Var → Maybe Var
danglingAnchor es x with lookupVar es x
... | nothing = nothing
... | just b with mode b
...   | owned      = nothing
...   | borrowed a with lookupVar es a
...     | nothing = just a
...     | just _  = nothing

-- Soundness: what the checker reports really is a dangling borrow.
danglingAnchor-sound :
  ∀ (es : Env) (x a : Var) → danglingAnchor es x ≡ just a → DanglingBorrow es x
danglingAnchor-sound es x a rep with lookupVar es x in lx
... | nothing = ⊥-elim (bad rep)
  where bad : nothing ≡ just a → ⊥
        bad ()
... | just b with mode b in mb
...   | owned = ⊥-elim (bad rep)
  where bad : nothing ≡ just a → ⊥
        bad ()
...   | borrowed a' with lookupVar es a' in la
...     | just _  = ⊥-elim (bad rep)
  where bad : nothing ≡ just a → ⊥
        bad ()
...     | nothing = dangling a' (entity b , is-b) la
  where
    -- Binding is a record, so `b` is definitionally `bind (entity b) (mode b)`
    -- -- eta. That is what lets the equation about `mode b` be transported into
    -- an equation about the whole binding without destructing it.
    is-b : lookupVar es x ≡ just (bind (entity b) (borrowed a'))
    is-b = trans lx (cong (λ md → just (bind (entity b) md)) mb)

------------------------------------------------------------------------
-- What a `borrow` step introduces, and what it does not.
--
-- The rule's premise is that the anchor is bound, so a FRESH borrow is never
-- dangling. Every dangling borrow therefore arises later, when the anchor goes
-- away -- which is the shape a checker should look for, and it means the check
-- belongs at the point the anchor dies rather than at the borrow.

fresh-borrow-is-not-dangling :
  ∀ (es : Env) (src dst : Var) (o : ObjId) (md : Mode) →
  sameVar dst src ≡ false →
  lookupVar es src ≡ just (bind o md) →
  danglingAnchor (bindVar es dst (bind o (borrowed src))) dst ≡ nothing
fresh-borrow-is-not-dangling es src dst o md ne look
  rewrite lookup-bind-here es dst (bind o (borrowed src))
        | lookup-bind-there es dst src (bind o (borrowed src)) ne
        | look = refl

-- And the step really does bind it that way. Without this the lemma above would
-- be about an environment the relation never produces.
-- Stated with the object taken FROM the step's own premise rather than supplied
-- separately: given independently they are two different `o`s and the equation
-- is not the one the rule proves.
borrow-step-binds-the-anchor :
  ∀ {f t bid rest es m src dst o md s'} →
  (st : f ⊢ pstate t bid (borrow dst src ∷ rest) es m —→ᵢ s') →
  lookupVar es src ≡ just (bind o md) →
  Σ ObjId λ o' → env s' ≡ bindVar es dst (bind o' (borrowed src))
borrow-step-binds-the-anchor {o = o} (step-borrow {o = o'} _) _ = o' , refl

------------------------------------------------------------------------
-- Dropping an anchor is what creates the danger.
--
-- `drop x` unbinds `x`. Any borrow anchored at `x` is dangling from that point
-- on. This is the theorem that says WHERE the check has to run: not at the
-- borrow, at the drop.

drop-strands-its-borrows :
  ∀ (es : Env) (x y : Var) (o : ObjId) →
  lookupVar es y ≡ just (bind o (borrowed x)) →
  sameVar y x ≡ false →
  lookupVar (unbindVar es x) x ≡ nothing →
  lookupVar (unbindVar es x) y ≡ just (bind o (borrowed x)) →
  DanglingBorrow (unbindVar es x) y
drop-strands-its-borrows es x y o look ne gone still =
  dangling x (o , still) gone

-- The hypothesis `lookupVar (unbindVar es x) x ≡ nothing` is not free: `bindVar`
-- shadows, so an environment with two bindings for `x` keeps one after
-- `unbindVar`. Requiring it rather than assuming it is the difference between a
-- theorem about this model and a theorem about a model where names are unique.

------------------------------------------------------------------------
-- Completeness.
--
-- The other direction, and the one that makes running the checker WORTH
-- anything: every dangling borrow is reported. Sound alone means "never cries
-- wolf" and is satisfied by a checker that reports nothing at all.
--
-- The proof is short because the specification and the procedure look at the
-- same two lookups. That is not luck -- it is why the anchor was put into
-- `Mode.borrowed` rather than left to be recovered by an analysis.

danglingAnchor-complete :
  ∀ (es : Env) (x : Var) → (d : DanglingBorrow es x) →
  danglingAnchor es x ≡ just (anchor d)
danglingAnchor-complete es x d with is-borrow d
... | (o , look) rewrite look | anchor-gone d = refl

-- Put together: the checker reports exactly the dangling borrows.
--
-- Stated as the two implications rather than as an `iff`, because they are used
-- in different places -- soundness by whoever acts on a report, completeness by
-- whoever concludes from silence that there is nothing to act on. The second is
-- the one a pass relies on when it elides a check.
danglingAnchor-exact :
  ∀ (es : Env) (x : Var) →
  ((a : Var) → danglingAnchor es x ≡ just a → DanglingBorrow es x)
  × ((d : DanglingBorrow es x) → danglingAnchor es x ≡ just (anchor d))
danglingAnchor-exact es x = danglingAnchor-sound es x , danglingAnchor-complete es x

-- And the contrapositive, which is the form a pass actually uses: silence means
-- there is nothing there.
silence-means-safe :
  ∀ (es : Env) (x : Var) →
  danglingAnchor es x ≡ nothing → ¬ DanglingBorrow es x
silence-means-safe es x quiet d with trans (sym quiet) (danglingAnchor-complete es x d)
... | ()

------------------------------------------------------------------------
-- Still-named, and premature reclaim.
--
-- This is the one that maps onto the compiler's open defects. A normal-path
-- double dealloc and an unattributed leak are both this shape: storage handed
-- back while a name still denotes what lived in it.

private
  no-just : ∀ {A : Set} {v : A} → nothing ≡ just v → ⊥
  no-just ()

just-inj′ : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
just-inj′ refl = refl

-- Hand-rolled rather than the standard library's membership, which is set up
-- over a setoid and drags in machinery these lemmas do not need.
data _∈ᵥ_ (x : Var) : List Var → Set where
  at-head : ∀ {xs} → x ∈ᵥ (x ∷ xs)
  in-tail : ∀ {y xs} → x ∈ᵥ xs → x ∈ᵥ (y ∷ xs)

data _∈ᵒ_ (o : ObjId) : List ObjId → Set where
  o-head : ∀ {os} → o ∈ᵒ (o ∷ os)
  o-tail : ∀ {q os} → o ∈ᵒ os → o ∈ᵒ (q ∷ os)

names : Env → List Var
names = map proj₁

private
  bound-is-a-key : ∀ (es : Env) (x : Var) (b : Binding) →
                   lookupVar es x ≡ just b → x ∈ᵥ names es
  bound-is-a-key [] x b ()
  bound-is-a-key ((y , c) ∷ es) x b look with sameVar y x in eq
  ... | true  = subst (_∈ᵥ (y ∷ names es)) (sameVar-sound y x eq) at-head
  ... | false = in-tail (bound-is-a-key es x b look)

  -- Inverting `entityOf`. This is the step that forced `entityOf` to be written
  -- with `map` rather than with a `with`: through an auxiliary function the
  -- equation does not reduce when the caller splits on the lookup.
  entityOf-bound : ∀ (es : Env) (x : Var) (o : ObjId) →
                   entityOf es x ≡ just o → Σ Binding λ b → lookupVar es x ≡ just b
  entityOf-bound es x o ent with lookupVar es x
  ... | just b  = b , refl
  ... | nothing = ⊥-elim (no-just ent)

-- "This name denotes this object", as a boolean. Tested through `entityOf`
-- rather than by reading the association list directly, and that is
-- load-bearing: binding SHADOWS, so an entry whose entity matches may be one
-- `lookupVar` never reaches. A checker that walked the entries would report a
-- name that does not denote the object, and soundness would be false.
denotes : Env → Var → ObjId → Bool
denotes es x o = maybe′ (λ p → sameObj p o) false (entityOf es x)

private
  denotes-sound : ∀ (es : Env) (x : Var) (o : ObjId) →
                  denotes es x o ≡ true → entityOf es x ≡ just o
  denotes-sound es x o d with entityOf es x
  ... | just p  = cong just (sameObj-sound p o d)
  ... | nothing with d
  ...   | ()

  denotes-complete : ∀ (es : Env) (x : Var) (o : ObjId) →
                     entityOf es x ≡ just o → denotes es x o ≡ true
  denotes-complete es x o ent with entityOf es x
  ... | just p  = subst (λ v → sameObj p v ≡ true) (just-inj′ ent) (sameObj-refl p)
  ... | nothing = ⊥-elim (no-just ent)

namedAmong : Env → List Var → ObjId → Maybe Var
namedAmong es []       o = nothing
namedAmong es (x ∷ xs) o = if denotes es x o then just x else namedAmong es xs o

stillNamed? : Env → ObjId → Maybe Var
stillNamed? es o = namedAmong es (names es) o

private
  namedAmong-sound : ∀ (es : Env) (xs : List Var) (o : ObjId) (x : Var) →
                     namedAmong es xs o ≡ just x → entityOf es x ≡ just o
  namedAmong-sound es []       o x ()
  namedAmong-sound es (y ∷ xs) o x rep with denotes es y o in dy
  ... | true  = subst (λ v → entityOf es v ≡ just o) (just-inj′ rep)
                      (denotes-sound es y o dy)
  ... | false = namedAmong-sound es xs o x rep

  namedAmong-finds : ∀ (es : Env) (xs : List Var) (o : ObjId) (x : Var) →
                     x ∈ᵥ xs → entityOf es x ≡ just o →
                     Σ Var λ z → namedAmong es xs o ≡ just z
  namedAmong-finds es (y ∷ xs) o x at-head ent with denotes es y o in dy
  ... | true  = y , refl
  ... | false with trans (sym (denotes-complete es y o ent)) dy
  ...   | ()
  namedAmong-finds es (y ∷ xs) o x (in-tail mem) ent with denotes es y o
  ... | true  = y , refl
  ... | false = namedAmong-finds es xs o x mem ent

-- Soundness: what it reports really does denote the object.
stillNamed?-sound : ∀ (es : Env) (o : ObjId) (x : Var) →
                    stillNamed? es o ≡ just x → StillNamed es o
stillNamed?-sound es o x rep = named-by x (namedAmong-sound es (names es) o x rep)

-- Completeness: if any name denotes it, the procedure reports one. NOT
-- necessarily the same one -- `StillNamed` carries an arbitrary witness and the
-- procedure returns the first key it finds -- so this is stated existentially,
-- which is the honest form and is what the contrapositive below needs.
stillNamed?-complete : ∀ (es : Env) (o : ObjId) →
                       StillNamed es o → Σ Var λ z → stillNamed? es o ≡ just z
stillNamed?-complete es o s =
  namedAmong-finds es (names es) o (name s)
    (bound-is-a-key es (name s) (proj₁ bnd) (proj₂ bnd)) (holds-it s)
  where
    bnd : Σ Binding λ b → lookupVar es (name s) ≡ just b
    bnd = entityOf-bound es (name s) o (holds-it s)

-- ⭐ The form a pass uses: silence means the object really is unnamed, so
-- handing its storage back is not premature.
unnamed-when-silent : ∀ (es : Env) (o : ObjId) →
                      stillNamed? es o ≡ nothing → ¬ StillNamed es o
unnamed-when-silent es o quiet s with stillNamed?-complete es o s
... | (z , rep) with trans (sym quiet) rep
...   | ()

------------------------------------------------------------------------
-- Premature reclaim.

isDead : Maybe Life → Bool
isDead (just dead)       = true
isDead (just live)       = false
isDead (just finalizing) = false
isDead nothing           = false

private
  isDead-sound : ∀ (ml : Maybe Life) → isDead ml ≡ true → ml ≡ just dead
  isDead-sound (just dead)       _  = refl
  isDead-sound (just live)       ()
  isDead-sound (just finalizing) ()
  isDead-sound nothing           ()

-- Written with `if` rather than with a `with`, for the reason this module keeps
-- running into: a `with` here would put the test inside an auxiliary function
-- and the soundness proof could not connect the branch it took to the fact it
-- needs about `lifeOf`.
prematureReclaim? : Env → Machine → ObjId → Maybe Var
prematureReclaim? es m o =
  if isDead (lifeOf m o) then stillNamed? es o else nothing

prematureReclaim?-sound :
  ∀ (es : Env) (m : Machine) (o : ObjId) (x : Var) →
  prematureReclaim? es m o ≡ just x → PrematureReclaim es m o
prematureReclaim?-sound es m o x rep with isDead (lifeOf m o) in dd
... | true  = premature (stillNamed?-sound es o x rep) (isDead-sound (lifeOf m o) dd)
... | false = ⊥-elim (no-just rep)

-- `pr` is destructured rather than projected, and that is not style: with-
-- abstraction rewrites the types of CONTEXT VARIABLES, so `bf` reduces under
-- the split while a freshly written `being-freed pr` would not, and the three
-- impossible branches could not be closed.
prematureReclaim?-complete :
  ∀ (es : Env) (m : Machine) (o : ObjId) →
  PrematureReclaim es m o → Σ Var λ z → prematureReclaim? es m o ≡ just z
prematureReclaim?-complete es m o (premature sn bf) with lifeOf m o
... | just dead = stillNamed?-complete es o sn
... | just live with bf
...   | ()
prematureReclaim?-complete es m o (premature sn bf) | just finalizing with bf
...   | ()
prematureReclaim?-complete es m o (premature sn bf) | nothing with bf
...   | ()

-- and the contrapositive, which is what licenses the free.
reclaim-is-safe-when-silent :
  ∀ (es : Env) (m : Machine) (o : ObjId) →
  prematureReclaim? es m o ≡ nothing → ¬ PrematureReclaim es m o
reclaim-is-safe-when-silent es m o quiet pr
  with prematureReclaim?-complete es m o pr
... | (z , rep) with trans (sym quiet) rep
...   | ()

------------------------------------------------------------------------
-- Leaks.
--
-- Both numbers are already computable, so the check is a comparison. What made
-- it unavailable before was not the arithmetic -- it was that `ownedCount` and
-- `ghostRC` had to branch on the SAME boolean before they could be compared at
-- all, which is why `ownedCount` is written with `isOwned … ∧ sameObj …`.

positive? : ℕ → Bool
positive? zero    = false
positive? (suc _) = true

zero? : ℕ → Bool
zero? zero    = true
zero? (suc _) = false

leaked? : Env → Machine → ObjId → Bool
leaked? es m o = positive? (ghostRC m o) ∧ zero? (ownedCount es o)

private
  positive?-sound : ∀ n → positive? n ≡ true → 0 < n
  positive?-sound (suc _) _ = s≤s z≤n

  positive?-complete : ∀ n → 0 < n → positive? n ≡ true
  positive?-complete (suc _) _ = refl

  zero?-sound : ∀ n → zero? n ≡ true → n ≡ 0
  zero?-sound zero _ = refl

  zero?-complete : ∀ n → n ≡ 0 → zero? n ≡ true
  zero?-complete zero _ = refl

  ∧-true : ∀ {a b : Bool} → (a ∧ b) ≡ true → (a ≡ true) × (b ≡ true)
  ∧-true {true} {true} _ = refl , refl

leaked?-sound : ∀ (es : Env) (m : Machine) (o : ObjId) →
                leaked? es m o ≡ true → Leaked es m o
leaked?-sound es m o rep with ∧-true rep
... | (gp , on) = leaked (positive?-sound (ghostRC m o) gp)
                         (zero?-sound (ownedCount es o) on)

leaked?-complete : ∀ (es : Env) (m : Machine) (o : ObjId) →
                   Leaked es m o → leaked? es m o ≡ true
leaked?-complete es m o lk
  rewrite positive?-complete (ghostRC m o) (still-owned lk)
        | zero?-complete (ownedCount es o) (unnamed lk) = refl

-- ⭐ The form a pass uses: if the check is silent at every object the pass
-- touched, nothing was leaked there.
no-leak-when-silent : ∀ (es : Env) (m : Machine) (o : ObjId) →
                      leaked? es m o ≡ false → ¬ Leaked es m o
no-leak-when-silent es m o quiet lk with trans (sym quiet) (leaked?-complete es m o lk)
... | ()

------------------------------------------------------------------------
-- The third invalidity: when a refcount update has to be atomic.
--
-- The check needs NO escape analysis, and that is the payoff of indexing owner
-- sites by thread from the start: "two threads can reach this object" is a walk
-- over the site map. `field′`, `global` and `queue` belong to no thread, so a
-- reference parked in one of them is unrelated to every other site -- which is
-- the escaped case, and it falls out rather than being special-cased.

sitesHolding : SiteMap → ObjId → List OwnerSite
sitesHolding []             o = []
sitesHolding ((s , p) ∷ ss) o =
  if sameObj p o then s ∷ sitesHolding ss o else sitesHolding ss o

-- Two sites belong to the same thread. `nothing` on either side answers false:
-- a site with no owning thread is reachable from ALL of them, so it is never
-- "the same thread" as anything.
sameOwnerThread : OwnerSite → OwnerSite → Bool
sameOwnerThread s u =
  maybe′ (λ a → maybe′ (λ b → a ≡ᵇ b) false (ownerThread u)) false (ownerThread s)

-- Two sites that cannot witness sharing: the same site, or the same thread.
related : OwnerSite → OwnerSite → Bool
related s u = sameSite s u ∨ sameOwnerThread s u

firstUnrelated : OwnerSite → List OwnerSite → Maybe OwnerSite
firstUnrelated s []       = nothing
firstUnrelated s (u ∷ us) = if related s u then firstUnrelated s us else just u

findShared : List OwnerSite → Maybe (OwnerSite × OwnerSite)
findShared []       = nothing
findShared (s ∷ us) = maybe′ (λ u → just (s , u)) (findShared us) (firstUnrelated s us)

sharedPair : Machine → ObjId → Maybe (OwnerSite × OwnerSite)
sharedPair m o = findShared (sitesHolding (sites m) o)

------------------------------------------------------------------------
-- Soundness.
--
-- Each equation about the three procedures is hoisted out rather than being
-- derived inside a `with`. `sitesHolding` and `firstUnrelated` both branch on a
-- boolean the caller cannot see through, and carrying the equation is what lets
-- a membership witness be transported across the branch.

private
  data _∈ˢ_ (s : OwnerSite) : List OwnerSite → Set where
    s-head : ∀ {us} → s ∈ˢ (s ∷ us)
    s-tail : ∀ {u us} → s ∈ˢ us → s ∈ˢ (u ∷ us)

  bad-bool : true ≡ false → ⊥
  bad-bool ()

  holding-hit : ∀ u p (ss : SiteMap) o → sameObj p o ≡ true →
                sitesHolding ((u , p) ∷ ss) o ≡ u ∷ sitesHolding ss o
  holding-hit u p ss o e rewrite e = refl

  holding-miss : ∀ u p (ss : SiteMap) o → sameObj p o ≡ false →
                 sitesHolding ((u , p) ∷ ss) o ≡ sitesHolding ss o
  holding-miss u p ss o e rewrite e = refl

  unrelated-skip : ∀ s v (us : List OwnerSite) → related s v ≡ true →
                   firstUnrelated s (v ∷ us) ≡ firstUnrelated s us
  unrelated-skip s v us e rewrite e = refl

  unrelated-take : ∀ s v (us : List OwnerSite) → related s v ≡ false →
                   firstUnrelated s (v ∷ us) ≡ just v
  unrelated-take s v us e rewrite e = refl

  shared-take : ∀ v w (us : List OwnerSite) → firstUnrelated v us ≡ just w →
                findShared (v ∷ us) ≡ just (v , w)
  shared-take v w us e rewrite e = refl

  shared-skip : ∀ v (us : List OwnerSite) → firstUnrelated v us ≡ nothing →
                findShared (v ∷ us) ≡ findShared us
  shared-skip v us e rewrite e = refl

  pair-inj : ∀ {A B : Set} {a c : A} {b d : B} → (a , b) ≡ (c , d) → (a ≡ c) × (b ≡ d)
  pair-inj refl = refl , refl

  holding-sound : ∀ (ss : SiteMap) (o : ObjId) (s : OwnerSite) →
                  s ∈ˢ sitesHolding ss o → Holds ss s o
  holding-sound []             o s ()
  holding-sound ((u , p) ∷ ss) o s mem = go (sameObj p o) refl
    where
      -- `q` is abstracted so that `s-head` can refine it. Matched against the
      -- outer `s` -- a parameter of the enclosing function -- it could not.
      branch : ∀ (e : sameObj p o ≡ true) q →
               q ∈ˢ (u ∷ sitesHolding ss o) → Holds ((u , p) ∷ ss) q o
      branch e .u s-head      = subst (Holds ((u , p) ∷ ss) u) (sameObj-sound p o e)
                                      holds-here
      branch e q  (s-tail mt) = holds-there (holding-sound ss o q mt)

      go : (bb : Bool) → sameObj p o ≡ bb → Holds ((u , p) ∷ ss) s o
      go true  e = branch e s (subst (s ∈ˢ_) (holding-hit u p ss o e) mem)
      go false e = holds-there
                     (holding-sound ss o s (subst (s ∈ˢ_) (holding-miss u p ss o e) mem))

  unrelated-distinct : ∀ (s u : OwnerSite) → related s u ≡ false → s ≢ u
  unrelated-distinct s u ne = go (sameSite s u) refl
    where
      go : (bb : Bool) → sameSite s u ≡ bb → s ≢ u
      go true  e with trans (sym (cong (λ z → z ∨ sameOwnerThread s u) e)) ne
      ... | ()
      go false e = sameSite-false e

  unrelated-threads :
    ∀ (s u : OwnerSite) → related s u ≡ false →
    ¬ (Σ ThreadId λ t → (ownerThread s ≡ just t) × (ownerThread u ≡ just t))
  unrelated-threads s u ne (t , (ps , pu)) = go (sameSite s u) refl
    where
      same : sameOwnerThread s u ≡ true
      same rewrite ps | pu | ≡ᵇ-refl t = refl

      go : (bb : Bool) → sameSite s u ≡ bb → ⊥
      go true  e with trans (sym (cong (λ z → z ∨ sameOwnerThread s u) e)) ne
      ... | ()
      go false e with trans (sym (trans (cong (λ z → z ∨ sameOwnerThread s u) e) same)) ne
      ... | ()

  firstUnrelated-sound :
    ∀ (s : OwnerSite) (us : List OwnerSite) (u : OwnerSite) →
    firstUnrelated s us ≡ just u → (u ∈ˢ us) × (related s u ≡ false)
  firstUnrelated-sound s []       u ()
  firstUnrelated-sound s (v ∷ us) u rep = go (related s v) refl
    where
      go : (bb : Bool) → related s v ≡ bb → (u ∈ˢ (v ∷ us)) × (related s u ≡ false)
      go true e =
        let (mem , unrel) = firstUnrelated-sound s us u
                              (trans (sym (unrelated-skip s v us e)) rep)
        in s-tail mem , unrel
      go false e =
        subst (λ z → (z ∈ˢ (v ∷ us)) × (related s z ≡ false))
              (just-inj′ (trans (sym (unrelated-take s v us e)) rep))
              (s-head , e)

  findShared-sound :
    ∀ (us : List OwnerSite) (s u : OwnerSite) →
    findShared us ≡ just (s , u) →
    (s ∈ˢ us) × (u ∈ˢ us) × (related s u ≡ false)
  findShared-sound []       s u ()
  findShared-sound (v ∷ us) s u rep = go (firstUnrelated v us) refl
    where
      go : (r : Maybe OwnerSite) → firstUnrelated v us ≡ r →
           (s ∈ˢ (v ∷ us)) × (u ∈ˢ (v ∷ us)) × (related s u ≡ false)
      go (just w) e =
        let (sv , uw)     = pair-inj (just-inj′ (trans (sym (shared-take v w us e)) rep))
            (mem , unrel) = firstUnrelated-sound v us w e
        in subst (_∈ˢ (v ∷ us)) sv s-head
         , subst (_∈ˢ (v ∷ us)) uw (s-tail mem)
         , subst₂ (λ a b → related a b ≡ false) sv uw unrel
      go nothing e =
        let (ms , mu , unrel) = findShared-sound us s u
                                  (trans (sym (shared-skip v us e)) rep)
        in s-tail ms , s-tail mu , unrel

-- ⭐ What the checker reports really is an object two threads can reach.
sharedPair-sound :
  ∀ (m : Machine) (o : ObjId) (s u : OwnerSite) →
  sharedPair m o ≡ just (s , u) → SharedAcrossThreads m o
sharedPair-sound m o s u rep =
  let (ms , mu , unrel) = findShared-sound (sitesHolding (sites m) o) s u rep
  in shared-by s u
       (holding-sound (sites m) o s ms)
       (holding-sound (sites m) o u mu)
       (unrelated-distinct s u unrel)
       (unrelated-threads s u unrel)

------------------------------------------------------------------------
-- Whether a refcount update on this object has to be atomic.

isCountedNow : Maybe RuntimeCount → Bool
isCountedNow (just (counted _)) = true
isCountedNow (just immortal)    = false
isCountedNow nothing            = false

needsAtomic? : Machine → ObjId → Maybe (OwnerSite × OwnerSite)
needsAtomic? m o = if isCountedNow (countOf m o) then sharedPair m o else nothing

private
  isCountedNow-sound : ∀ (mc : Maybe RuntimeCount) → isCountedNow mc ≡ true →
                       Σ ℕ λ n → mc ≡ just (counted n)
  isCountedNow-sound (just (counted n)) _ = n , refl

isCountedNow-complete : ∀ (mc : Maybe RuntimeCount) (n : ℕ) → mc ≡ just (counted n) →
                        isCountedNow mc ≡ true
isCountedNow-complete (just (counted _)) n refl = refl

needsAtomic?-sound :
  ∀ (m : Machine) (o : ObjId) (s u : OwnerSite) →
  needsAtomic? m o ≡ just (s , u) → NeedsAtomicRC m o
needsAtomic?-sound m o s u rep = go (isCountedNow (countOf m o)) refl
  where
    go : (bb : Bool) → isCountedNow (countOf m o) ≡ bb → NeedsAtomicRC m o
    go true e = sharedPair-sound m o s u
                  (trans (sym (step e)) rep) , isCountedNow-sound (countOf m o) e
      where step : isCountedNow (countOf m o) ≡ true →
                   needsAtomic? m o ≡ sharedPair m o
            step ee rewrite ee = refl
    go false e = ⊥-elim (no-just (trans (sym (step e)) rep))
      where step : isCountedNow (countOf m o) ≡ false → needsAtomic? m o ≡ nothing
            step ee rewrite ee = refl

-- ⭐ And an immortal is never reported, whatever the site map says. The
-- theorem that licenses NOT emitting an atomic there.
immortal-needs-nothing :
  ∀ (m : Machine) (o : ObjId) → countOf m o ≡ just immortal →
  needsAtomic? m o ≡ nothing
immortal-needs-nothing m o imm rewrite imm = refl

------------------------------------------------------------------------
-- Completeness of the sharing check.
--
-- The direction that makes silence mean something. Without it `needsAtomic?`
-- could be `λ _ _ → nothing` and every soundness proof above would still hold.

private
  related-refl : ∀ (s : OwnerSite) → related s s ≡ true
  related-refl s rewrite sameSite-refl s = refl

  sameOwnerThread-sym : ∀ (s u : OwnerSite) →
                        sameOwnerThread s u ≡ sameOwnerThread u s
  sameOwnerThread-sym s u with ownerThread s | ownerThread u
  ... | just a  | just b  = ≡ᵇ-sym a b
  ... | just _  | nothing = refl
  ... | nothing | just _  = refl
  ... | nothing | nothing = refl

  related-sym : ∀ (s u : OwnerSite) → related s u ≡ related u s
  related-sym s u rewrite sameSite-sym s u | sameOwnerThread-sym s u = refl

  -- Everything the scan passed over really was related.
  firstUnrelated-nothing :
    ∀ (s : OwnerSite) (us : List OwnerSite) (u : OwnerSite) →
    firstUnrelated s us ≡ nothing → u ∈ˢ us → related s u ≡ true
  firstUnrelated-nothing s []       u _    ()
  firstUnrelated-nothing s (v ∷ us) u none mem = go (related s v) refl
    where
      branch : related s v ≡ true → ∀ q → q ∈ˢ (v ∷ us) → related s q ≡ true
      branch e .v s-head      = e
      branch e q  (s-tail mt) =
        firstUnrelated-nothing s us q (trans (sym (unrelated-skip s v us e)) none) mt

      go : (bb : Bool) → related s v ≡ bb → related s u ≡ true
      go true  e = branch e u mem
      go false e = ⊥-elim (just-not-nothing (trans (sym (unrelated-take s v us e)) none))
        where just-not-nothing : ∀ {A : Set} {w : A} → just w ≡ nothing → ⊥
              just-not-nothing ()

  holding-complete : ∀ (ss : SiteMap) (o : ObjId) (s : OwnerSite) →
                     Holds ss s o → s ∈ˢ sitesHolding ss o
  holding-complete _ o s holds-here rewrite sameObj-refl o = s-head
  holding-complete ((u , p) ∷ ss) o s (holds-there h) = go (sameObj p o) refl
    where
      go : (bb : Bool) → sameObj p o ≡ bb → s ∈ˢ sitesHolding ((u , p) ∷ ss) o
      go true  e = subst (s ∈ˢ_) (sym (holding-hit u p ss o e))
                         (s-tail (holding-complete ss o s h))
      go false e = subst (s ∈ˢ_) (sym (holding-miss u p ss o e))
                         (holding-complete ss o s h)

  -- ⭐ A list containing two mutually unrelated sites makes the scan report.
  --
  -- The hypothesis travels WITH the two sites rather than being read from the
  -- enclosing clause: abstracting them so the membership patterns can refine
  -- would otherwise leave `related s u ≡ false` talking about two other sites.
  findShared-finds :
    ∀ (us : List OwnerSite) (s u : OwnerSite) →
    s ∈ˢ us → u ∈ˢ us → related s u ≡ false →
    Σ OwnerSite λ a → Σ OwnerSite λ b → findShared us ≡ just (a , b)
  findShared-finds []       s u () _ _
  findShared-finds (v ∷ us) s u ms mu unrel = go (firstUnrelated v us) refl
    where
      go : (r : Maybe OwnerSite) → firstUnrelated v us ≡ r →
           Σ OwnerSite λ a → Σ OwnerSite λ b → findShared (v ∷ us) ≡ just (a , b)
      go (just w) e = v , w , shared-take v w us e
      go nothing  e = pick s ms u mu unrel
        where
          all-related : ∀ q → q ∈ˢ us → related v q ≡ true
          all-related q mm = firstUnrelated-nothing v us q e mm

          pick : ∀ a → a ∈ˢ (v ∷ us) → ∀ b → b ∈ˢ (v ∷ us) → related a b ≡ false →
                 Σ OwnerSite λ x → Σ OwnerSite λ y →
                   findShared (v ∷ us) ≡ just (x , y)
          pick .v s-head .v s-head ur =
            ⊥-elim (bad-bool (trans (sym (related-refl v)) ur))
          pick .v s-head b (s-tail mt) ur =
            ⊥-elim (bad-bool (trans (sym (all-related b mt)) ur))
          pick a (s-tail mt) .v s-head ur =
            ⊥-elim (bad-bool (trans (sym (all-related a mt))
                                    (trans (sym (related-sym a v)) ur)))
          pick a (s-tail mta) b (s-tail mtb) ur =
            let (x , y , eq) = findShared-finds us a b mta mtb ur
            in x , y , trans (shared-skip v us e) eq

  same-site-false : ∀ (s u : OwnerSite) → s ≢ u → sameSite s u ≡ false
  same-site-false s u ne = go (sameSite s u) refl
    where
      go : (bb : Bool) → sameSite s u ≡ bb → sameSite s u ≡ false
      go true  e = ⊥-elim (ne (sameSite-sound s u e))
      go false e = e

  same-thread-false :
    ∀ (s u : OwnerSite) →
    ¬ (Σ ThreadId λ t → (ownerThread s ≡ just t) × (ownerThread u ≡ just t)) →
    sameOwnerThread s u ≡ false
  same-thread-false s u none with ownerThread s | ownerThread u
  ... | just a  | just b  = go (a ≡ᵇ b) refl
    where
      go : (bb : Bool) → (a ≡ᵇ b) ≡ bb → (a ≡ᵇ b) ≡ false
      go true  e = ⊥-elim (none (a , refl , cong just (sym (≡ᵇ-sound a b e))))
      go false e = e
  ... | just _  | nothing = refl
  ... | nothing | just _  = refl
  ... | nothing | nothing = refl

sharedPair-complete :
  ∀ (m : Machine) (o : ObjId) → SharedAcrossThreads m o →
  Σ OwnerSite λ a → Σ OwnerSite λ b → sharedPair m o ≡ just (a , b)
sharedPair-complete m o sh =
  findShared-finds (sitesHolding (sites m) o) (site₁ sh) (site₂ sh)
    (holding-complete (sites m) o (site₁ sh) (holds₁ sh))
    (holding-complete (sites m) o (site₂ sh) (holds₂ sh))
    unrel
  where
    unrel : related (site₁ sh) (site₂ sh) ≡ false
    unrel rewrite same-site-false (site₁ sh) (site₂ sh) (distinct sh)
                | same-thread-false (site₁ sh) (site₂ sh) (not-same-thread sh) = refl

-- ⭐ Every object that needs an atomic refcount update is reported.
needsAtomic?-complete :
  ∀ (m : Machine) (o : ObjId) → NeedsAtomicRC m o →
  Σ OwnerSite λ a → Σ OwnerSite λ b → needsAtomic? m o ≡ just (a , b)
needsAtomic?-complete m o (sh , (n , cnt))
  rewrite isCountedNow-complete (countOf m o) n cnt = sharedPair-complete m o sh

-- and the contrapositive, which is the form that licenses NOT emitting one.
plain-is-safe-when-silent :
  ∀ (m : Machine) (o : ObjId) →
  needsAtomic? m o ≡ nothing → ¬ NeedsAtomicRC m o
plain-is-safe-when-silent m o quiet na
  with needsAtomic?-complete m o na
... | (a , b , rep) with trans (sym quiet) rep
...   | ()

------------------------------------------------------------------------
-- Finite domains for the checks.
--
-- `AllChecksSilent` in Proof.Lython.Decide used to quantify over `Var` and
-- `ObjId`, both of which are ℕ -- so discharging it meant proving something
-- about every natural number rather than running a finite check. A compiler has
-- the finite list; the model did not.

objectsOwned : SiteMap → List ObjId
objectsOwned = map proj₂

objectsNamed : Env → List ObjId
objectsNamed = map (λ b → entity (proj₂ b))

owned-is-listed : ∀ (ss : SiteMap) (o : ObjId) → 0 < logicalRC ss o → o ∈ᵒ objectsOwned ss
owned-is-listed []             o ()
owned-is-listed ((s , q) ∷ ss) o pos = go (sameObj q o) refl
  where
    miss : sameObj q o ≡ false → logicalRC ((s , q) ∷ ss) o ≡ logicalRC ss o
    miss e rewrite e = refl

    go : (bb : Bool) → sameObj q o ≡ bb → o ∈ᵒ (q ∷ objectsOwned ss)
    go true  e = subst (λ z → z ∈ᵒ (q ∷ objectsOwned ss)) (sameObj-sound q o e) o-head
    go false e = o-tail (owned-is-listed ss o (subst (0 <_) (miss e) pos))

holds-is-listed : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                  Holds ss s o → o ∈ᵒ objectsOwned ss
holds-is-listed ss s o h = owned-is-listed ss o (holds-positive ss s o h)

named-is-listed : ∀ (es : Env) (x : Var) (o : ObjId) →
                  entityOf es x ≡ just o → o ∈ᵒ objectsNamed es
named-is-listed []             x o ent = ⊥-elim (no-just ent)
named-is-listed ((u , b) ∷ es) x o ent = go (sameVar u x) refl
  where
    hit : sameVar u x ≡ true → entityOf ((u , b) ∷ es) x ≡ just (entity b)
    hit e rewrite e = refl
    miss : sameVar u x ≡ false → entityOf ((u , b) ∷ es) x ≡ entityOf es x
    miss e rewrite e = refl

    go : (bb : Bool) → sameVar u x ≡ bb → o ∈ᵒ (entity b ∷ objectsNamed es)
    go true  e = subst (λ z → z ∈ᵒ (entity b ∷ objectsNamed es))
                       (just-inj′ (trans (sym (hit e)) ent)) o-head
    go false e = o-tail (named-is-listed es x o (trans (sym (miss e)) ent))

name-is-listed : ∀ (es : Env) (x : Var) (b : Binding) →
                 lookupVar es x ≡ just b → x ∈ᵥ names es
name-is-listed = bound-is-a-key

------------------------------------------------------------------------
-- Running a check over a list.

data Every {A : Set} (P : A → Set) : List A → Set where
  every-nil  : Every P []
  every-cons : ∀ {y ys} → P y → Every P ys → Every P (y ∷ ys)

every-at-var : ∀ {P : Var → Set} {xs} → Every P xs → ∀ {x} → x ∈ᵥ xs → P x
every-at-var (every-cons px _) at-head     = px
every-at-var (every-cons _ ps) (in-tail m) = every-at-var ps m

every-at-obj : ∀ {P : ObjId → Set} {os} → Every P os → ∀ {o} → o ∈ᵒ os → P o
every-at-obj (every-cons po _) o-head     = po
every-at-obj (every-cons _ ps) (o-tail m) = every-at-obj ps m
