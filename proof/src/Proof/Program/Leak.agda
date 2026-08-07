{-# OPTIONS --safe #-}

-- ⭐ A leak, tied to reachability.
--
-- `Proof.Program.Coherence` had seven witnesses and said so plainly: "these are
-- seven witnesses, not a preservation theorem". `Leaked` was definable and
-- `coherent-has-no-leaks` turned coherence into leak-freedom, but nothing said
-- that RUNNING a program keeps a state coherent -- so leak-freedom held of the
-- seven states someone had written down and no others. That is the same hole
-- `WFRC` had before `Proof.Program.Preservation`, in the one place it matters
-- most: the suite's leak stage exists because at least seven goldens were green
-- while leaking, and a leak is exactly the defect no crash and no fuzzer finds.
--
-- The theorem is `no-reachable-state-leaks`, and it is NOT "the compiler is
-- leak-free" -- it is not. Coherence is preserved by every rule, so no sequence
-- of instructions can produce a leak; a leak therefore needs a boundary the
-- relation does not have. The two of those and the caveat are recorded at the
-- theorem itself rather than twice.
--
-- Why coherence and not `WFRC`: `WFRC.live-positive` says a live counted object
-- has a positive ghost count -- owner sites exist. It says nothing about NAMES,
-- and a leak is precisely sites without names. The two invariants are about the
-- two sides of the same equation and neither implies the other.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Leak (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false; if_then_else_; _∧_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing; maybe′)
open import Data.Nat using (ℕ; zero; suc; _+_; _<_; s≤s; z≤n)
open import Data.Nat.Properties using (+-suc)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; subst; subst₂)
open import Relation.Nullary using (¬_)

open import Proof.RC.Object using (ObjId; sameObj; sameObj-sound; sameObj-refl)
open import Proof.RC.OwnerSite using (OwnerSite; ThreadId; SiteMap; occupy; vacate;
  unnamedRC; isUnnamedSite; field′; unnamedRC-occupy-named;
  unnamedRC-vacate-named;
  strongAt; logicalRC; vacate-holder; vacate-holder-other; callee;
  unnamedRC-vacate-holder; unnamedRC-vacate-holder-other)
open import Proof.RC.Machine Sig using (Machine; machine; heap; objects; sites;
  ghostRC)
open import Proof.Program.Syntax using (Var; Function; Instr; params)
open import Proof.Program.Env
open import Proof.Program.Step Sig
open import Proof.Program.Preservation Sig using (WF; WFES; backed;
  moveOne-preserves; step-preserves-WF)
open import Proof.Lython.Invalid Sig using (Leaked; still-owned; unnamed; unheld)

------------------------------------------------------------------------
-- The property.
--
-- Quantified over ALL objects, not just the one in play. A version stated at a
-- single object would be satisfied by a machine that leaked every other one,
-- and leaks are precisely what the owned-name side is there to see.

-- ⭐ Owned NAMES plus occupied FIELDS, against the count.
--
-- The name half alone was the invariant while the model had no field rule, and
-- it is false the moment there is one: a store moves a hold from a name to a
-- field, so the name side drops by one while the count does not move. Both
-- halves together are what the runtime counter actually counts.
NameSiteCoherent : Env → Machine → Set
NameSiteCoherent es m = ∀ o → ownedCount es o + unnamedRC (sites m) o ≡ ghostRC m o

Coherent : PState → Set
Coherent s = NameSiteCoherent (env s) (mach s)

-- ⭐ Coherence is exactly leak-freedom.
--
-- `Leaked es m o` says the count is positive while NOTHING holds it -- no name
-- and no field. Coherence rules that out for every object at once, which is
-- what the ∀-quantification is for: both holding sides add up to the count, so
-- if both are zero the count is zero.
--
-- The field conjunct is what keeps this true once fields can hold. An aggregate
-- member has no name and is not leaked; the parent's release vacates its field.
coherent-has-no-leaks :
  ∀ (es : Env) (m : Machine) → NameSiteCoherent es m → ∀ o → ¬ Leaked es m o
coherent-has-no-leaks es m coh o lk
  with subst (0 <_) (sym zeroed) (still-owned lk)
  where
    -- `0 + k` is `k` by reduction, so dropping the name half leaves the field
    -- half, and dropping that leaves nothing for the count to be.
    named-gone : ownedCount es o + unnamedRC (sites m) o ≡ unnamedRC (sites m) o
    named-gone = cong (_+ unnamedRC (sites m) o) (unnamed lk)

    zeroed : 0 ≡ ghostRC m o
    zeroed = trans (sym (unheld lk)) (trans (sym named-gone) (coh o))
... | ()

------------------------------------------------------------------------
-- One shape for both counts.
--
-- ⭐ This is the payoff of writing `ownedCount` as `if isOwned (mode b) ∧ … `
-- rather than with a `with` on the mode. Binding a name and occupying a site
-- are the SAME operation on a number -- add one when a boolean holds -- and
-- because both are spelled as the same `if`, one case split reduces both. A
-- `with` would have made them two auxiliary functions and no single split could
-- reach them together, which is the whole content of comparing the two counts.

bump : Bool → ℕ → ℕ
bump b n = if b then suc n else n

private
  suc-inj : ∀ {a b : ℕ} → suc a ≡ suc b → a ≡ b
  suc-inj refl = refl

  just-inj : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
  just-inj refl = refl

  bump-inj : ∀ (b : Bool) (u v : ℕ) → bump b u ≡ bump b v → u ≡ v
  bump-inj true  u v e = suc-inj e
  bump-inj false u v e = e

  -- Two independent increments commute. This is the whole content of the
  -- inductive step below, isolated so that the induction does not have to carry
  -- a four-way case split through it.
  bump-swap : ∀ (b₁ b₂ : Bool) (n : ℕ) →
              bump b₁ (bump b₂ n) ≡ bump b₂ (bump b₁ n)
  bump-swap true  true  n = refl
  bump-swap true  false n = refl
  bump-swap false true  n = refl
  bump-swap false false n = refl

  same→eq : ∀ (o p : ObjId) → sameObj o p ≡ true → p ≡ o
  same→eq o p e = sym (sameObj-sound o p e)

  -- The field half rides along. Every rule that bumps the name side leaves the
  -- field side alone, so the two-sided equation follows from the one-sided one
  -- by pushing the bump past an addition.
  bump-plus : ∀ (b : Bool) (n k : ℕ) → bump b n + k ≡ bump b (n + k)
  bump-plus true  n k = refl
  bump-plus false n k = refl

  -- And the mirror, for the one rule that bumps the FIELD side instead:
  -- `setField` takes a name off and puts a field on.
  bump-plusʳ : ∀ (b : Bool) (n k : ℕ) → n + bump b k ≡ bump b (n + k)
  bump-plusʳ true  n k = +-suc n k
  bump-plusʳ false n k = refl

  -- Occupying a site no name owns, in the shape the counts branch on. Once the
  -- site is quantified the `∧` no longer reduces on its own, which is what the
  -- premise is for.
  unnamedRC-occupy-unnamed : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) →
    isUnnamedSite s ≡ true →
    unnamedRC (occupy ss s o) p ≡ bump (sameObj o p) (unnamedRC ss p)
  unnamedRC-occupy-unnamed ss s o p un rewrite un = refl

-- Whether a binding contributes to an object's owned-name count. Named because
-- it appears in every statement below and because it is the boolean the site
-- side branches on too, once the binding is owned.
counts : Binding → ObjId → Bool
counts b o = isOwned (mode b) ∧ sameObj (entity b) o

-- Both sides, as `bump`. Both hold by computation; they are stated so the
-- rewriting below has a name to use rather than relying on the reader to see
-- that two definitions unfold to the same shape.
ownedCount-bind : ∀ (es : Env) (x : Var) (b : Binding) (o : ObjId) →
                  ownedCount (bindVar es x b) o ≡ bump (counts b o) (ownedCount es o)
ownedCount-bind es x b o = refl

logicalRC-occupy : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) →
                   logicalRC (occupy ss s o) p ≡ bump (sameObj o p) (logicalRC ss p)
logicalRC-occupy ss s o p = refl

-- An owned binding contributes on exactly the boolean the site side uses.
counts-owned : ∀ (o p : ObjId) → counts (bind o owned) p ≡ sameObj o p
counts-owned o p = refl

------------------------------------------------------------------------
-- Removing a name.
--
-- `lookupVar` returns the FIRST entry with the name and `unbindVar` removes the
-- first entry with the name, so the entry that goes is exactly the one the rule
-- looked up. That agreement is what makes this an equation rather than an
-- inequality, and it is why no SSA premise is needed here: a shadowed binding
-- is still counted, and still counted after the unbind.

ownedCount-unbind :
  ∀ (es : Env) (x : Var) (b : Binding) → lookupVar es x ≡ just b →
  ∀ o → ownedCount es o ≡ bump (counts b o) (ownedCount (unbindVar es x) o)
ownedCount-unbind []             x b ()   o
ownedCount-unbind ((y , c) ∷ es) x b look o = go (sameVar y x) refl
  where
    go : (bb : Bool) → sameVar y x ≡ bb →
         ownedCount ((y , c) ∷ es) o
           ≡ bump (counts b o) (ownedCount (unbindVar ((y , c) ∷ es) x) o)
    -- No `rewrite` on the boolean, and the reason is the one this development
    -- has hit repeatedly: it reduces the goal and leaves `look` talking about
    -- the unreduced `if`, so the two no longer meet. The cons equations carry
    -- the branch explicitly instead.
    go true e =
      trans (cong (λ z → bump (counts z o) (ownedCount es o))
                  (just-inj (trans (sym (lookupVar-cons-true y c es x e)) look)))
            (cong (λ E → bump (counts b o) (ownedCount E o))
                  (sym (unbindVar-cons-true y c es x e)))
    go false e =
      trans (cong (bump (counts c o))
                  (ownedCount-unbind es x b
                     (trans (sym (lookupVar-cons-false y c es x e)) look) o))
            (trans (bump-swap (counts c o) (counts b o) (ownedCount (unbindVar es x) o))
                   (cong (λ E → bump (counts b o) (ownedCount E o))
                         (sym (unbindVar-cons-false y c es x e))))

-- ⭐ The site-side twin. Vacating a site that HOLDS the object takes one off its
-- count and leaves every other object alone -- the two halves `Proof.RC.OwnerSite`
-- proves separately, joined into the one shape the comparison needs.
logicalRC-vacate :
  ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) → strongAt ss s ≡ just o →
  ∀ p → logicalRC ss p ≡ bump (sameObj o p) (logicalRC (vacate ss s) p)
logicalRC-vacate ss s o held p = go (sameObj o p) refl
  where
    go : (bb : Bool) → sameObj o p ≡ bb →
         logicalRC ss p ≡ bump (sameObj o p) (logicalRC (vacate ss s) p)
    go true  e rewrite e | same→eq o p e = vacate-holder ss s o held
    go false e rewrite e = sym (vacate-holder-other ss s o p held e)

-- The unnamed half of `logicalRC-vacate`. Only a site no name owns moves this
-- count, which is why it takes the premise the logical one does not.
unnamedRC-vacate :
  ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
  isUnnamedSite s ≡ true → strongAt ss s ≡ just o →
  ∀ p → unnamedRC ss p ≡ bump (sameObj o p) (unnamedRC (vacate ss s) p)
unnamedRC-vacate ss s o un held p = go (sameObj o p) refl
  where
    go : (bb : Bool) → sameObj o p ≡ bb →
         unnamedRC ss p ≡ bump (sameObj o p) (unnamedRC (vacate ss s) p)
    go true  e rewrite e | same→eq o p e = unnamedRC-vacate-holder ss s o un held
    go false e rewrite e = sym (unnamedRC-vacate-holder-other ss s o p held e)

------------------------------------------------------------------------
-- The rules.
--
-- Every one of them adds one to both counts, takes one off both, or touches
-- neither. That is the whole proof, and it is short because `bump` made the two
-- sides the same shape -- the work was in the two lemmas above, not here.
--
-- `WF` is a hypothesis and cannot be dropped: `logicalRC-vacate` needs to know
-- the site being vacated really held the object, and only `WFES.backed` turns
-- the rule's environment premise into that machine fact. A `vacate` of a site
-- that held something else would take the count off the wrong object.

-- The shared half of `move` and `drop`: after the unbind and the vacate, the
-- two counts still agree.
private
  after-removal :
    ∀ (t : ThreadId) (es : Env) (ss : SiteMap) (x : Var) (o : ObjId) →
    lookupVar es x ≡ just (bind o owned) →
    strongAt ss (siteOf t x) ≡ just o →
    (∀ p → ownedCount es p + unnamedRC ss p ≡ logicalRC ss p) →
    ∀ p → ownedCount (unbindVar es x) p + unnamedRC (vacate ss (siteOf t x)) p
            ≡ logicalRC (vacate ss (siteOf t x)) p
  after-removal t es ss x o look held coh p =
    bump-inj (sameObj o p) _ _
      (trans (cong (bump (sameObj o p))
                   (cong (ownedCount (unbindVar es x) p +_)
                         (unnamedRC-vacate-named ss (siteOf t x) p refl)))
      (trans (sym (bump-plus (sameObj o p) (ownedCount (unbindVar es x) p)
                             (unnamedRC ss p)))
      (trans (cong (_+ unnamedRC ss p)
                   (sym (ownedCount-unbind es x (bind o owned) look p)))
             (trans (coh p) (logicalRC-vacate ss (siteOf t x) o held p)))))

  -- ⭐ The dual of `after-removal`: a hold leaves the UNNAMED side instead of
  -- the name side. `drop` takes one off the name half and the count; `callIn`
  -- takes one off this half and the count, and the equation survives for the
  -- same reason -- both sides fall together.
  after-unnamed-removal :
    ∀ (es : Env) (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
    isUnnamedSite s ≡ true → strongAt ss s ≡ just o →
    (∀ p → ownedCount es p + unnamedRC ss p ≡ logicalRC ss p) →
    ∀ p → ownedCount es p + unnamedRC (vacate ss s) p
            ≡ logicalRC (vacate ss s) p
  after-unnamed-removal es ss s o un held coh p =
    bump-inj (sameObj o p) _ _
      (trans (sym (bump-plusʳ (sameObj o p) (ownedCount es p)
                              (unnamedRC (vacate ss s) p)))
      (trans (cong (ownedCount es p +_) (sym (unnamedRC-vacate ss s o un held p)))
             (trans (coh p) (logicalRC-vacate ss s o held p))))

-- Occupying a NAME's site: one more owned name, one more site, and the field
-- half untouched. Every rule that binds a name is this shape.
private
  keeps-fields :
    ∀ (t : ThreadId) (x : Var) (o : ObjId) (ss : SiteMap) (n : ℕ) (p : ObjId) →
    n + unnamedRC ss p ≡ logicalRC ss p →
    bump (sameObj o p) n + unnamedRC (occupy ss (siteOf t x) o) p
      ≡ bump (sameObj o p) (logicalRC ss p)
  keeps-fields t x o ss n p e =
    trans (cong (bump (sameObj o p) n +_)
                (unnamedRC-occupy-named ss (siteOf t x) o p refl))
    (trans (bump-plus (sameObj o p) n (unnamedRC ss p))
           (cong (bump (sameObj o p)) e))

instr-preserves-coherence : ∀ {f s u} → f ⊢ s —→ᵢ u → WF s → Coherent s → Coherent u
-- One more name, one more site, on the same boolean.
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-alloc {x = x} {o = o} _ _ _ _ _) w coh p =
  keeps-fields t x o (sites m) (ownedCount es p) p (coh p)
-- The header write touches neither side.
instr-preserves-coherence (step-init _ _ _)                 w coh p = coh p
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-dup {dst = dst} {o = o} _ _ _) w coh p =
  keeps-fields t dst o (sites m) (ownedCount es p) p (coh p)
-- One off, one on, at both levels.
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-move {src = src} {dst = dst} {o = o} look _) w coh p =
  keeps-fields t dst o (vacate (sites m) (siteOf t src))
               (ownedCount (unbindVar es src) p) p
               (after-removal t es (sites m) src o look (backed w src o look)
                              coh p)
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-drop {x = x} {o = o} look _ _ _) w coh p =
  after-removal t es (sites m) x o look (backed w x o look) coh p
-- A borrow is not counted, so `bump false` on the name side and nothing at all
-- on the site side. This is "a borrow costs nothing" read off the equation.
instr-preserves-coherence (step-borrow _)                   w coh p = coh p
-- ⭐ A store takes the name off and puts the FIELD on. The count does not move,
-- and the two halves of the invariant swap exactly one hold between them --
-- which is the whole reason the invariant has two halves.
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-set-field {src = src} {k = k} {p = r} {o = o} _ look nodup _) w coh q =
  trans (cong (ownedCount (unbindVar es src) q +_)
              (unnamedRC-occupy-unnamed (vacate (sites m) (siteOf t src))
                                        (field′ r k) o q refl))
  (trans (bump-plusʳ (sameObj o q) (ownedCount (unbindVar es src) q)
                     (unnamedRC (vacate (sites m) (siteOf t src)) q))
         (cong (bump (sameObj o q))
               (after-removal t es (sites m) src o look (backed w src o look)
                              coh q)))
-- A read binds a borrowed name and moves nothing.
instr-preserves-coherence (step-get-field _ _)              w coh p = coh p
-- ⭐ And receiving: the hold comes off the unnamed side and onto a name. The
-- exact reverse of `callOut`, discharged by the two halves separately -- the
-- removal, then `keeps-fields`, which is what every rule that binds a name uses.
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-call-in {dst = dst} {c = c} {o = o} at-src) w coh p =
  keeps-fields t dst o (vacate (sites m) (callee t c)) (ownedCount es p) p
               (after-unnamed-removal es (sites m) (callee t c) o refl at-src
                                      coh p)
-- ⭐ And a transfer at a call is the same swap `setField` is: the name side
-- gives up a hold and the unnamed side takes it. Word for word `setField`'s
-- proof with a different destination, which is what generalising `fieldRC`
-- bought -- with the field-only count there was no term for this hold to land
-- in and the equation simply broke.
instr-preserves-coherence {s = pstate t bid _ es m}
  (step-call-out {src = src} {c = c} {o = o} look nodup) w coh q =
  trans (cong (ownedCount (unbindVar es src) q +_)
              (unnamedRC-occupy-unnamed (vacate (sites m) (siteOf t src))
                                        (callee t c) o q refl))
  (trans (bump-plusʳ (sameObj o q) (ownedCount (unbindVar es src) q)
                     (unnamedRC (vacate (sites m) (siteOf t src)) q))
         (cong (bump (sameObj o q))
               (after-removal t es (sites m) src o look
                              (backed w src o look) coh q)))

------------------------------------------------------------------------
-- Block arguments.

moveOne-preserves-coherence :
  ∀ (t : ThreadId) (m : Machine) (es : Env) (ss : SiteMap) (p a : Var)
    (es' : Env) (ss' : SiteMap) →
  moveOne t (es , ss) p a ≡ just (es' , ss') →
  WFES t es (afterArgs m ss) →
  NameSiteCoherent es (afterArgs m ss) → NameSiteCoherent es' (afterArgs m ss')
moveOne-preserves-coherence t m es ss p a es' ss' eq w coh = outer (lookupVar es a) refl
  where
    tail-of : Binding → Maybe (Env × SiteMap)
    tail-of b = maybe′ (λ _ → nothing)
                       (just (bindVar (unbindVar es a) p b , relocate t ss a p b))
                       (lookupVar (unbindVar es a) a)

    result : ∀ (b : Binding) → lookupVar es a ≡ just b →
             NameSiteCoherent (bindVar (unbindVar es a) p b)
                              (afterArgs m (relocate t ss a p b))
    result (bind o owned) look q =
      keeps-fields t p o (vacate ss (siteOf t a))
                   (ownedCount (unbindVar es a) q) q
                   (after-removal t es ss a o look (backed w a o look) coh q)
    -- The borrowed case relocates NO site, and removes a name that was not
    -- counted -- so both sides are exactly where they were.
    result (bind o (borrowed v)) look q =
      trans (cong (_+ unnamedRC ss q)
                  (sym (ownedCount-unbind es a (bind o (borrowed v)) look q)))
            (coh q)

    middle : ∀ (b : Binding) → lookupVar es a ≡ just b →
             (r : Maybe Binding) → lookupVar (unbindVar es a) a ≡ r →
             maybe′ (λ _ → nothing)
                    (just (bindVar (unbindVar es a) p b , relocate t ss a p b))
                    r ≡ just (es' , ss') →
             NameSiteCoherent es' (afterArgs m ss')
    middle b look (just _) e h with h
    ...                          | ()
    middle b look nothing  e h =
      subst₂ (λ E S → NameSiteCoherent E (afterArgs m S))
             (cong proj₁ (just-inj h)) (cong proj₂ (just-inj h))
             (result b look)

    outer : (r : Maybe Binding) → lookupVar es a ≡ r →
            NameSiteCoherent es' (afterArgs m ss')
    outer nothing e
      with trans (sym (cong (maybe′ tail-of nothing) e)) eq
    ...  | ()
    outer (just b) e =
      middle b e (lookupVar (unbindVar es a) a) refl
             (trans (sym (cong (maybe′ tail-of nothing) e)) eq)

moveArgs-preserves-coherence :
  ∀ (t : ThreadId) (m : Machine) (es : Env) (ss : SiteMap) (ps as : List Var)
    (es' : Env) (ss' : SiteMap) →
  moveArgs t (es , ss) ps as ≡ just (es' , ss') →
  WFES t es (afterArgs m ss) →
  NameSiteCoherent es (afterArgs m ss) → NameSiteCoherent es' (afterArgs m ss')
moveArgs-preserves-coherence t m es ss [] [] es' ss' eq w coh =
  subst₂ (λ E S → NameSiteCoherent E (afterArgs m S))
         (cong proj₁ (just-inj eq)) (cong proj₂ (just-inj eq)) coh
moveArgs-preserves-coherence t m es ss (p ∷ ps) (a ∷ as) es' ss' eq w coh =
  go (moveOne t (es , ss) p a) refl
  where
    go : (r : Maybe (Env × SiteMap)) → moveOne t (es , ss) p a ≡ r →
         NameSiteCoherent es' (afterArgs m ss')
    go nothing e
      with trans (sym (cong (maybe′ (λ st' → moveArgs t st' ps as) nothing) e)) eq
    ...  | ()
    go (just (es₁ , ss₁)) e =
      moveArgs-preserves-coherence t m es₁ ss₁ ps as es' ss'
        (trans (sym (cong (maybe′ (λ st' → moveArgs t st' ps as) nothing) e)) eq)
        (moveOne-preserves t m es ss p a es₁ ss₁ e w)
        (moveOne-preserves-coherence t m es ss p a es₁ ss₁ e w coh)
moveArgs-preserves-coherence t m es ss []      (_ ∷ _) es' ss' eq w coh with eq
...                                                                       | ()
moveArgs-preserves-coherence t m es ss (_ ∷ _) []      es' ss' eq w coh with eq
...                                                                       | ()

term-preserves-coherence : ∀ {f s u} → f ⊢ s —→ₜ u → WF s → Coherent s → Coherent u
term-preserves-coherence {s = pstate t bid [] es m}
  (step-br {args = args} {nxt = nxt} _ _ _ mv) w coh =
  moveArgs-preserves-coherence t m es (sites m) (params nxt) args _ _ mv w coh
term-preserves-coherence {s = pstate t bid [] es m}
  (step-cond-then {a₁ = a₁} {nxt = nxt} _ _ _ mv) w coh =
  moveArgs-preserves-coherence t m es (sites m) (params nxt) a₁ _ _ mv w coh
term-preserves-coherence {s = pstate t bid [] es m}
  (step-cond-else {a₂ = a₂} {nxt = nxt} _ _ _ mv) w coh =
  moveArgs-preserves-coherence t m es (sites m) (params nxt) a₂ _ _ mv w coh
term-preserves-coherence {s = pstate t bid [] es m}
  (step-invoke-normal {a = a} {nxt = nxt} _ _ _ mv) w coh =
  moveArgs-preserves-coherence t m es (sites m) (params nxt) a _ _ mv w coh
term-preserves-coherence {s = pstate t bid [] es m}
  (step-invoke-throw {pa = pa} {nxt = nxt} _ _ _ mv) w coh =
  moveArgs-preserves-coherence t m es (sites m) (params nxt) pa _ _ mv w coh

------------------------------------------------------------------------
-- ⭐ THE THEOREMS.

step-preserves-coherence : ∀ {f s u} → f ⊢ s —→ u → WF s → Coherent s → Coherent u
step-preserves-coherence (by-instr st) w coh = instr-preserves-coherence st w coh
step-preserves-coherence (by-term  st) w coh = term-preserves-coherence st w coh

reachable-preserves-coherence :
  ∀ {f s u} → f ⊢ s —→* u → WF s → Coherent s → Coherent u
reachable-preserves-coherence done        w coh = coh
reachable-preserves-coherence (more p ps) w coh =
  reachable-preserves-coherence ps (step-preserves-WF p w) (step-preserves-coherence p w coh)

-- ⭐ NOTHING REACHABLE FROM A COHERENT STATE LEAKS.
--
-- The statement the suite's leak stage was built to check, over the step
-- relation rather than over a list of witnesses.
--
-- Read the contrapositive for what it says about the compiler. A leak needs
-- either an incoherent START or a transition this relation does not have, and
-- there are exactly two of those: FUNCTION ENTRY (`invoke` moves operands to a
-- successor block of the same function -- there is no callee frame here) and
-- SCOPE EXIT (nothing discards an environment). So every leak is a boundary,
-- not a placement: no sequence of instructions produces one.
--
-- ⛔ That is not "this compiler does not leak" -- it does. It is a statement
-- about WHERE to look, and it points away from families A-E, which were all
-- placement. Whether the two boundaries above are where the open leak families
-- actually live has NOT been checked against the compiler; it is the prediction
-- this theorem makes, filed in REMAINING-GAPS.md rather than claimed here.
no-reachable-state-leaks :
  ∀ {f s u} → f ⊢ s —→* u → WF s → Coherent s →
  ∀ o → ¬ Leaked (env u) (mach u) o
no-reachable-state-leaks {u = u} r w coh =
  coherent-has-no-leaks (env u) (mach u) (reachable-preserves-coherence r w coh)
