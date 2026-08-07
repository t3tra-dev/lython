{-# OPTIONS --safe #-}

-- ⭐ Preservation: every step takes a well-formed state to a well-formed state.
--
-- This was the largest open item in the development. `WFRC` was stated, six
-- machines were shown to satisfy it, and nothing said that RUNNING a program
-- keeps it satisfied -- so the invariant constrained the states someone had
-- written down and no others.
--
-- Three things had to change before it could be proved, and each is recorded
-- where it happened rather than here:
--
--   * `WFRC.no-stale-owner` was stated over `strongAt`, which reports the FIRST
--     entry at a site. `vacate` removes that entry and exposes the next, about
--     which a first-entry property says nothing. It is now stated over `Holds`
--     -- membership -- which is both preservable and the property one wants.
--
--   * Five rules gained premises. Each is a legality condition whose absence
--     lets the rule produce a state violating the invariant: `dup` of a dead
--     object is a resurrection, `new` onto absent storage is a dangling
--     reference at birth, and `move`/`drop` on a shadowed name leave the
--     environment owning a reference the site map no longer records.
--
--   * A BLOCK ARGUMENT BECAME A MOVE. With `br` leaving the operand bound, the
--     invariant was false after any branch with arguments and no theorem over
--     `—→*` could be stated at all.
--
-- The state invariant is bigger than `WFRC`, and it has to be. `WFRC` alone is
-- not preserved by `move`: the rule vacates `siteOf src` and nothing in `WFRC`
-- says that site held anything, so the count could fall by one. What licenses
-- the vacate is that the ENVIRONMENT and the SITE MAP agree -- `backed` below.

open import Proof.Memory.Element using (ElemSig)

module Proof.Program.Preservation (Sig : ElemSig) where

open ElemSig Sig

open import Data.Bool using (Bool; true; false)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_)
open import Data.Maybe using (Maybe; just; nothing; maybe′)
import Data.Maybe
open import Data.Nat using (ℕ; zero; suc; _<_; s≤s; z≤n)
open import Data.Product using (_×_; _,_; Σ; proj₁; proj₂)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong; subst; subst₂)
open import Relation.Nullary using (¬_)

open import Proof.Memory.Heap using (Heap; Block; lookupBlock; generation;
  liveness)
  renaming (live to blockLive)
open import Proof.Memory.Descriptor Sig using (Desc)
open import Proof.RC.Object
open import Proof.RC.OwnerSite
open import Proof.RC.Machine Sig
open import Proof.RC.Invariant Sig
open import Proof.RC.Properties Sig using (lookupObj-cons-true;
  lookupObj-cons-false; updateObj-cons-true; updateObj-cons-false;
  lookupObj-update-same)
open import Proof.Program.Syntax
  using (Var; Function; Instr; alloc; init; move; dup; drop; borrow; params)
open import Proof.Program.Env
open import Proof.Program.Step Sig

private
  just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
  just-inj refl = refl

  counted-inj : ∀ {a b : ℕ} → counted a ≡ counted b → a ≡ b
  counted-inj refl = refl

  suc-inj : ∀ {a b : ℕ} → suc a ≡ suc b → a ≡ b
  suc-inj refl = refl

------------------------------------------------------------------------
-- Table and environment lemmas the rules need.

-- The companion of `lookupObj-update-same`. Updating one object leaves every
-- other one's entry alone -- which is what makes "this rule touched exactly one
-- object" a statement the invariant can use.
lookupObj-update-other :
  ∀ (ts : ObjTable) (o p : ObjId) (f : ObjCell → ObjCell) →
  sameObj o p ≡ false →
  lookupObj (updateObj ts o f) p ≡ lookupObj ts p
lookupObj-update-other []             o p f ne = refl
lookupObj-update-other ((q , c) ∷ ts) o p f ne = head-case (sameObj q o) refl
  where
    tail-case : (bb : Bool) → sameObj q p ≡ bb →
                lookupObj ((q , c) ∷ updateObj ts o f) p ≡ lookupObj ((q , c) ∷ ts) p
    tail-case true qp =
      trans (lookupObj-cons-true q c (updateObj ts o f) p qp)
            (sym (lookupObj-cons-true q c ts p qp))
    tail-case false qp =
      trans (lookupObj-cons-false q c (updateObj ts o f) p qp)
            (trans (lookupObj-update-other ts o p f ne)
                   (sym (lookupObj-cons-false q c ts p qp)))

    head-case : (bb : Bool) → sameObj q o ≡ bb →
                lookupObj (updateObj ((q , c) ∷ ts) o f) p ≡ lookupObj ((q , c) ∷ ts) p
    head-case true qo =
      trans (cong (λ z → lookupObj z p) (updateObj-cons-true q c ts o f qo))
            (trans (lookupObj-cons-false q (f c) ts p qp)
                   (sym (lookupObj-cons-false q c ts p qp)))
      where
        -- `q ≡ o` and `o` is not `p`, so this entry is not `p`'s either.
        qp : sameObj q p ≡ false
        qp = trans (cong (λ z → sameObj z p) (sameObj-sound q o qo)) ne
    head-case false qo =
      trans (cong (λ z → lookupObj z p) (updateObj-cons-false q c ts o f qo))
            (tail-case (sameObj q p) refl)

-- Unbinding one name leaves every other name's binding alone. Needed because
-- `move` and `drop` unbind their source and the invariant then has to be
-- re-established for all the OTHER names.
unbindVar-other :
  ∀ (es : Env) (x y : Var) → sameVar x y ≡ false →
  lookupVar (unbindVar es x) y ≡ lookupVar es y
unbindVar-other []             x y ne = refl
unbindVar-other ((u , b) ∷ es) x y ne = head-case (sameVar u x) refl
  where
    tail-case : (bb : Bool) → sameVar u y ≡ bb →
                lookupVar ((u , b) ∷ unbindVar es x) y ≡ lookupVar ((u , b) ∷ es) y
    tail-case true uy =
      trans (lookupVar-cons-true u b (unbindVar es x) y uy)
            (sym (lookupVar-cons-true u b es y uy))
    tail-case false uy =
      trans (lookupVar-cons-false u b (unbindVar es x) y uy)
            (trans (unbindVar-other es x y ne)
                   (sym (lookupVar-cons-false u b es y uy)))

    head-case : (bb : Bool) → sameVar u x ≡ bb →
                lookupVar (unbindVar ((u , b) ∷ es) x) y ≡ lookupVar ((u , b) ∷ es) y
    head-case true ux =
      trans (cong (λ z → lookupVar z y) (unbindVar-cons-true u b es x ux))
            (sym (lookupVar-cons-false u b es y uy))
      where
        uy : sameVar u y ≡ false
        uy = trans (cong (λ z → sameVar z y) (sameVar-sound u x ux)) ne
    head-case false ux =
      trans (cong (λ z → lookupVar z y) (unbindVar-cons-false u b es x ux))
            (tail-case (sameVar u y) refl)

-- Two names of one thread occupy the same site exactly when they are the same
-- name. This is where the program layer's notion of identity meets the site
-- map's, and it is why `siteOf` had to become thread-indexed: with a fixed
-- thread the two threads' slot 3 were the same site.
sameSite-siteOf : ∀ (t : ThreadId) (x y : Var) →
                  sameSite (siteOf t x) (siteOf t y) ≡ sameVar x y
sameSite-siteOf t x y rewrite ≡ᵇ-refl t = refl

------------------------------------------------------------------------
-- The state invariant.

-- Indexed by the three things it is about rather than by a `PState`, so that
-- the lemmas below can be applied to an environment and a machine that no state
-- currently holds -- which is what the argument-moving recursion needs.
record WFES (t : ThreadId) (es : Env) (m : Machine) : Set where
  constructor wfes
  field
    rc : WFRC m
    -- ⭐ Every OWNED name occupies its own site, holding its own entity.
    --
    -- This is the bridge, and without it no rule that consults the environment
    -- can be shown to do the right thing to the machine. `step-move` knows
    -- `lookupVar es src ≡ just (bind o owned)` -- an environment fact -- and
    -- vacates `siteOf src` -- a machine action. Only `backed` connects them.
    --
    -- Borrowed names are deliberately excluded: a borrow occupies no site, and
    -- requiring one here would make "a borrow costs nothing" false.
    backed : ∀ x o → lookupVar es x ≡ just (bind o owned) →
             strongAt (sites m) (siteOf t x) ≡ just o

open WFES public

WF : PState → Set
WF s = WFES (onThread s) (env s) (mach s)

-- Kept as an alias so the constructor reads the same at use sites.
wfs : ∀ {t es m} → WFRC m →
      (∀ x o → lookupVar es x ≡ just (bind o owned) →
               strongAt (sites m) (siteOf t x) ≡ just o) →
      WFES t es m
wfs = wfes

------------------------------------------------------------------------
-- Reading a machine after a step.

private
  -- Explicit arguments throughout. `sameObj` pattern matches on both of its
  -- arguments, so a metavariable in either position blocks reduction and the
  -- unifier cannot recover the table or the key from the equation alone.
  count-from : ∀ (ts : ObjTable) (o : ObjId) (c : ObjCell) →
               lookupObj ts o ≡ just c →
               Data.Maybe.map count (lookupObj ts o) ≡ just (count c)
  count-from ts o c e = cong (Data.Maybe.map count) e

  life-from : ∀ (ts : ObjTable) (o : ObjId) (c : ObjCell) →
              lookupObj ts o ≡ just c →
              Data.Maybe.map life (lookupObj ts o) ≡ just (life c)
  life-from ts o c e = cong (Data.Maybe.map life) e

  count-eq : ∀ (ts us : ObjTable) (o : ObjId) →
             lookupObj ts o ≡ lookupObj us o →
             Data.Maybe.map count (lookupObj ts o)
               ≡ Data.Maybe.map count (lookupObj us o)
  count-eq ts us o e = cong (Data.Maybe.map count) e

  life-eq : ∀ (ts us : ObjTable) (o : ObjId) →
            lookupObj ts o ≡ lookupObj us o →
            Data.Maybe.map life (lookupObj ts o)
              ≡ Data.Maybe.map life (lookupObj us o)
  life-eq ts us o e = cong (Data.Maybe.map life) e

  same→eq : ∀ (o p : ObjId) → sameObj o p ≡ true → p ≡ o
  same→eq o p e = sym (sameObj-sound o p e)

------------------------------------------------------------------------
-- `borrow`: the machine is untouched, so only `backed` has anything to prove.
--
-- Stated over `WFES` rather than over two `PState`s: `WF` does not mention the
-- block id or the pending instructions, so quantifying over them here would
-- leave them unsolvable at the call site.

-- Binding a name as BORROWED changes nothing the invariant is about, and the
-- source binding is no part of the argument: the new name can never be the
-- owned binding `backed` describes, which is the whole proof. `borrow` and
-- `getField` differ only in where the object came from -- a name for one, a
-- field site for the other -- so neither is mentioned here and both use this.
borrowed-bind-preserves : ∀ {t es m anchor dst c} →
  WFES t es m →
  WFES t (bindVar es dst (bind c (borrowed anchor))) m
borrowed-bind-preserves {t = t} {es = es} {m = m} {anchor = src} {dst = dst} {c = o} w =
  wfs (rc w) bk
  where
    bk : ∀ y p → lookupVar (bindVar es dst (bind o (borrowed src))) y
                   ≡ just (bind p owned) →
         strongAt (sites m) (siteOf t y) ≡ just p
    bk y p h with sameVar dst y
    -- The new name is BORROWED, so it can never be the owned binding the
    -- hypothesis describes. That is the whole content of this case.
    ... | true  with h
    ...   | ()
    bk y p h | false = backed w y p h

private
  bind-obj : ∀ {o₁ o₂ md₁ md₂} → bind o₁ md₁ ≡ bind o₂ md₂ → o₁ ≡ o₂
  bind-obj refl = refl

------------------------------------------------------------------------
-- `alloc`: storage and a name, and no object yet.
--
-- A parameterised module rather than a `where` block: every field of the
-- invariant needs the same five premises, and threading them through five
-- signatures obscured which fact each field used.
--
-- Every field about a counter or a life is discharged by CONTRADICTION at the
-- fresh id, because `lookupObj` answers `nothing` there. That is the model of
-- the initialisation window: the storage exists and is owned, and there is no
-- object to have a refcount.

module Alloc
  {t : ThreadId} {es : Env} {m : Machine} {x : Var} {o : ObjId} {b : Block}
  (fresh   : lookupObj (objects m) o ≡ nothing)
  (unowned : logicalRC (sites m) o ≡ 0)
  (blk     : lookupBlock (heap m) (objAllocation o) ≡ just b)
  (gen     : generation b ≡ objGeneration o)
  (alv     : liveness b ≡ blockLive)
  (w       : WFES t es m)
  where

  private
    ss' : SiteMap
    ss' = occupy (sites m) (siteOf t x) o

    m' : Machine
    m' = machine (heap m) (objects m) ss'

    no-cell : ∀ p → sameObj o p ≡ true → lookupObj (objects m) p ≡ nothing
    no-cell p e rewrite same→eq o p e = fresh

    ghost-hit : ∀ p → sameObj o p ≡ true → ghostRC m' p ≡ 1
    ghost-hit p e rewrite same→eq o p e =
      trans (occupy-same (sites m) (siteOf t x) o) (cong suc unowned)

    ghost-miss : ∀ p → sameObj o p ≡ false → ghostRC m' p ≡ ghostRC m p
    ghost-miss p e = occupy-other (sites m) (siteOf t x) o p e

  preserves : WFES t (bindVar es x (bind o owned)) m'
  preserves = wfs (record { counted-exact = ce
                          ; live-positive = lp
                          ; dead-unowned = du
                          ; no-stale-owner = ns
                          ; owned-storage-live = osl }) bk'
    where
      ce : ∀ p n → countOf m' p ≡ just (counted n) → lifeOf m' p ≡ just live →
           n ≡ ghostRC m' p
      ce p n cnt lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → n ≡ ghostRC m' p
          go true  e with trans (sym (cong (Data.Maybe.map count) (no-cell p e))) cnt
          ...           | ()
          go false e = trans (counted-exact (rc w) p n cnt lif) (sym (ghost-miss p e))

      lp : ∀ p → lifeOf m' p ≡ just live → IsCounted m' p → 0 < ghostRC m' p
      lp p lif cnt = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → 0 < ghostRC m' p
          go true  e with trans (sym (cong (Data.Maybe.map life) (no-cell p e))) lif
          ...           | ()
          go false e = subst (0 <_) (sym (ghost-miss p e)) (live-positive (rc w) p lif cnt)

      du : ∀ p → lifeOf m' p ≡ just dead → ghostRC m' p ≡ 0
      du p lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → ghostRC m' p ≡ 0
          go true  e with trans (sym (cong (Data.Maybe.map life) (no-cell p e))) lif
          ...           | ()
          go false e = trans (ghost-miss p e) (dead-unowned (rc w) p lif)

      -- The new site is the only one the rule adds, and the three heap premises
      -- are exactly what it needs. This is why they sit on `alloc` rather than
      -- on `init`: the site exists from here.
      ns : ∀ s p → Holds ss' s p →
           Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                    × (generation bb ≡ objGeneration p)
      ns s p holds-here      = b , blk , gen
      ns s p (holds-there h) = no-stale-owner (rc w) s p h

      osl : ∀ p → 0 < ghostRC m' p →
            Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                     × (liveness bb ≡ blockLive)
      osl p pos = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb →
               Σ _ λ bb′ → (lookupBlock (heap m) (objAllocation p) ≡ just bb′)
                         × (liveness bb′ ≡ blockLive)
          go true  e rewrite same→eq o p e = b , blk , alv
          go false e = owned-storage-live (rc w) p (subst (0 <_) (ghost-miss p e) pos)

      bk' : ∀ y q → lookupVar (bindVar es x (bind o owned)) y ≡ just (bind q owned) →
            strongAt ss' (siteOf t y) ≡ just q
      bk' y q h = go (sameVar x y) refl
        where
          go : (bb : Bool) → sameVar x y ≡ bb → strongAt ss' (siteOf t y) ≡ just q
          go true e =
            trans (strongAt-cons-true (siteOf t x) o (sites m) (siteOf t y)
                     (trans (sameSite-siteOf t x y) e))
                  (cong just (bind-obj (just-inj
                     (trans (sym (lookupVar-cons-true x (bind o owned) es y e)) h))))
          go false e =
            trans (strongAt-cons-false (siteOf t x) o (sites m) (siteOf t y)
                     (trans (sameSite-siteOf t x y) e))
                  (backed w y q
                     (trans (sym (lookupVar-cons-false x (bind o owned) es y e)) h))

------------------------------------------------------------------------
-- `init`: the header write.
--
-- The site map and the environment are untouched, so `no-stale-owner`,
-- `owned-storage-live` and `backed` transfer unchanged -- the site this object
-- occupies was vetted by `alloc` and nothing has happened to it since. The only
-- work is `counted-exact` at the new cell, and `alone` is exactly what it needs:
-- the counter is written as 1, so one site had better be holding it.

module Init
  {t : ThreadId} {es : Env} {m : Machine} {x : Var} {o : ObjId} {bk : Desc 1}
  (look  : lookupVar es x ≡ just (bind o owned))
  (fresh : lookupObj (objects m) o ≡ nothing)
  (alone : logicalRC (sites m) o ≡ 1)
  (w     : WFES t es m)
  where

  private
    born : ObjCell
    born = cell live (counted 1) bk 0

    ts' : ObjTable
    ts' = (o , born) ∷ objects m

    m' : Machine
    m' = machine (heap m) ts' (sites m)

    count-hit : ∀ p → sameObj o p ≡ true → countOf m' p ≡ just (counted 1)
    count-hit p e = count-from ts' p born (lookupObj-cons-true o born (objects m) p e)

    life-hit : ∀ p → sameObj o p ≡ true → lifeOf m' p ≡ just live
    life-hit p e = life-from ts' p born (lookupObj-cons-true o born (objects m) p e)

    tbl-miss : ∀ p → sameObj o p ≡ false → lookupObj ts' p ≡ lookupObj (objects m) p
    tbl-miss p e = lookupObj-cons-false o born (objects m) p e

    count-miss : ∀ p → sameObj o p ≡ false → countOf m' p ≡ countOf m p
    count-miss p e = count-eq ts' (objects m) p (tbl-miss p e)

    life-miss : ∀ p → sameObj o p ≡ false → lifeOf m' p ≡ lifeOf m p
    life-miss p e = life-eq ts' (objects m) p (tbl-miss p e)

    ghost-hit : ∀ p → sameObj o p ≡ true → ghostRC m' p ≡ 1
    ghost-hit p e rewrite same→eq o p e = alone

  preserves : WFES t es m'
  preserves = wfs (record { counted-exact = ce
                          ; live-positive = lp
                          ; dead-unowned = du
                          ; no-stale-owner = no-stale-owner (rc w)
                          ; owned-storage-live = owned-storage-live (rc w) }) (backed w)
    where
      ce : ∀ p n → countOf m' p ≡ just (counted n) → lifeOf m' p ≡ just live →
           n ≡ ghostRC m' p
      ce p n cnt lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → n ≡ ghostRC m' p
          go true  e = trans (counted-inj (just-inj (trans (sym cnt) (count-hit p e))))
                             (sym (ghost-hit p e))
          go false e = counted-exact (rc w) p n
                         (trans (sym (count-miss p e)) cnt)
                         (trans (sym (life-miss p e)) lif)

      lp : ∀ p → lifeOf m' p ≡ just live → IsCounted m' p → 0 < ghostRC m' p
      lp p lif (n , cnt) = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → 0 < ghostRC m' p
          go true  e = subst (0 <_) (sym (ghost-hit p e)) (s≤s z≤n)
          go false e = live-positive (rc w) p (trans (sym (life-miss p e)) lif)
                         (n , trans (sym (count-miss p e)) cnt)

      du : ∀ p → lifeOf m' p ≡ just dead → ghostRC m' p ≡ 0
      du p lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → ghostRC m' p ≡ 0
          go true  e with trans (sym (life-hit p e)) lif
          ...           | ()
          go false e = dead-unowned (rc w) p (trans (sym (life-miss p e)) lif)

------------------------------------------------------------------------
-- `dup` = py.incref. Counter up, one more site, one more name.
--
-- The heap premises `new` carries are absent here and do not need to be: the
-- object is already owned by `src`, so `backed` turns the environment fact into
-- `Holds (sites m) (siteOf t src) o` and the invariant's own fields supply the
-- block. This is the first place `backed` pays for itself.

module Dup
  {t : ThreadId} {es : Env} {m : Machine} {src dst : Var} {o : ObjId} {c : ObjCell}
  (look : lookupVar es src ≡ just (bind o owned))
  (tbl  : lookupObj (objects m) o ≡ just c)
  (alive : life c ≡ live)
  (w    : WFES t es m)
  where

  private
    up : ObjCell → ObjCell
    up y = record y { count = bumpUp (count y) }

    ts' : ObjTable
    ts' = updateObj (objects m) o up

    ss' : SiteMap
    ss' = occupy (sites m) (siteOf t dst) o

    m' : Machine
    m' = machine (heap m) ts' ss'

    -- `src` names it, so the map records it. Everything about the heap follows.
    held : Holds (sites m) (siteOf t src) o
    held = strongAt-holds (sites m) (siteOf t src) o (backed w src o look)

    positive : 0 < ghostRC m o
    positive = holds-positive (sites m) (siteOf t src) o held

    hit-cell : lookupObj ts' o ≡ just (up c)
    hit-cell = lookupObj-update-same (objects m) o up c tbl

    count-hit : countOf m' o ≡ just (bumpUp (count c))
    count-hit = count-from ts' o (up c) hit-cell

    life-hit : lifeOf m' o ≡ just (life c)
    life-hit = life-from ts' o (up c) hit-cell

    tbl-miss : ∀ p → sameObj o p ≡ false → lookupObj ts' p ≡ lookupObj (objects m) p
    tbl-miss p e = lookupObj-update-other (objects m) o p up e

    count-miss : ∀ p → sameObj o p ≡ false → countOf m' p ≡ countOf m p
    count-miss p e = count-eq ts' (objects m) p (tbl-miss p e)

    life-miss : ∀ p → sameObj o p ≡ false → lifeOf m' p ≡ lifeOf m p
    life-miss p e = life-eq ts' (objects m) p (tbl-miss p e)

    ghost-hit : ghostRC m' o ≡ suc (ghostRC m o)
    ghost-hit = occupy-same (sites m) (siteOf t dst) o

    ghost-miss : ∀ p → sameObj o p ≡ false → ghostRC m' p ≡ ghostRC m p
    ghost-miss p e = occupy-other (sites m) (siteOf t dst) o p e

  preserves : WFES t (bindVar es dst (bind o owned)) m'
  preserves = wfs (record { counted-exact = ce
                          ; live-positive = lp
                          ; dead-unowned = du
                          ; no-stale-owner = ns
                          ; owned-storage-live = osl }) bk'
    where
      -- The interesting half. `bumpUp` on a counted cell is `suc`, and the site
      -- map grew by exactly one entry, so the two sides move together. On an
      -- immortal cell there is nothing to prove -- and that is not a gap: an
      -- immortal object's counter is deliberately not tracking its sites.
      ce-hit : ∀ n → countOf m' o ≡ just (counted n) → n ≡ ghostRC m' o
      ce-hit n cnt = go (count c) refl
        where
          go : (r : RuntimeCount) → count c ≡ r → n ≡ ghostRC m' o
          go (counted k) e =
            trans (counted-inj (just-inj
                    (trans (sym cnt) (trans count-hit (cong (λ r → just (bumpUp r)) e)))))
                  (trans (cong suc prior) (sym ghost-hit))
            where
              prior : k ≡ ghostRC m o
              prior = counted-exact (rc w) o k
                        (trans (count-from (objects m) o c tbl) (cong just e))
                        (trans (life-from (objects m) o c tbl) (cong just alive))
          go immortal e with trans (sym cnt) (trans count-hit (cong (λ r → just (bumpUp r)) e))
          ...              | ()

      ce : ∀ p n → countOf m' p ≡ just (counted n) → lifeOf m' p ≡ just live →
           n ≡ ghostRC m' p
      ce p n cnt lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → n ≡ ghostRC m' p
          go true  e rewrite same→eq o p e = ce-hit n cnt
          go false e = trans (counted-exact (rc w) p n
                               (trans (sym (count-miss p e)) cnt)
                               (trans (sym (life-miss p e)) lif))
                             (sym (ghost-miss p e))

      lp : ∀ p → lifeOf m' p ≡ just live → IsCounted m' p → 0 < ghostRC m' p
      lp p lif (n , cnt) = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → 0 < ghostRC m' p
          go true  e rewrite same→eq o p e = subst (0 <_) (sym ghost-hit) (s≤s z≤n)
          go false e = subst (0 <_) (sym (ghost-miss p e))
                         (live-positive (rc w) p (trans (sym (life-miss p e)) lif)
                           (n , trans (sym (count-miss p e)) cnt))

      du : ∀ p → lifeOf m' p ≡ just dead → ghostRC m' p ≡ 0
      du p lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → ghostRC m' p ≡ 0
          go true  e rewrite same→eq o p e
            with trans (sym (trans life-hit (cong just alive))) lif
          ...  | ()
          go false e = trans (ghost-miss p e)
                             (dead-unowned (rc w) p (trans (sym (life-miss p e)) lif))

      ns : ∀ s p → Holds ss' s p →
           Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                    × (generation bb ≡ objGeneration p)
      ns s p holds-here      = no-stale-owner (rc w) (siteOf t src) o held
      ns s p (holds-there h) = no-stale-owner (rc w) s p h

      osl : ∀ p → 0 < ghostRC m' p →
            Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                     × (liveness bb ≡ blockLive)
      osl p pos = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb →
               Σ _ λ bb′ → (lookupBlock (heap m) (objAllocation p) ≡ just bb′)
                         × (liveness bb′ ≡ blockLive)
          go true  e rewrite same→eq o p e = owned-storage-live (rc w) o positive
          go false e = owned-storage-live (rc w) p (subst (0 <_) (ghost-miss p e) pos)

      bk' : ∀ y q → lookupVar (bindVar es dst (bind o owned)) y ≡ just (bind q owned) →
            strongAt ss' (siteOf t y) ≡ just q
      bk' y q h = go (sameVar dst y) refl
        where
          go : (bb : Bool) → sameVar dst y ≡ bb → strongAt ss' (siteOf t y) ≡ just q
          go true e =
            trans (strongAt-cons-true (siteOf t dst) o (sites m) (siteOf t y)
                     (trans (sameSite-siteOf t dst y) e))
                  (cong just (bind-obj (just-inj
                     (trans (sym (lookupVar-cons-true dst (bind o owned) es y e)) h))))
          go false e =
            trans (strongAt-cons-false (siteOf t dst) o (sites m) (siteOf t y)
                     (trans (sameSite-siteOf t dst y) e))
                  (backed w y q
                     (trans (sym (lookupVar-cons-false dst (bind o owned) es y e)) h))

------------------------------------------------------------------------
-- `move`: the entity changes hands. Nothing about the object table changes and
-- -- the point of the rule -- neither does the ghost count: one site is vacated
-- and one occupied.
--
-- `vacate-holder` is the lemma that makes that true, and it needs to know the
-- source's site really held the object. Only `backed` supplies that; without it
-- the vacate could drop a count that was never there and `counted-exact` would
-- fail on a rule that does nothing to the counter.

-- The half that does not care WHERE the entity lands: vacate one site, occupy
-- another, and the ghost count is unmoved. `move` lands it on a name's site and
-- `setField` on a field's, and every field of the invariant below is proved the
-- same way for both -- so it is proved once, over the destination SITE.
-- The SOURCE is a site rather than a name, for the same reason the destination
-- already was. `look` only ever produced `at-src`, and a name is not the only
-- thing that can hold: `callIn` takes its reference off `callee t c`, which no
-- name is backed by.
module MoveCore
  {t : ThreadId} {es : Env} {m : Machine} {o : ObjId}
  (ssite  : OwnerSite)
  (dsite  : OwnerSite)
  (at-src : strongAt (sites m) ssite ≡ just o)
  (w      : WFES t es m)
  where

  vacated : SiteMap
  vacated = vacate (sites m) ssite

  ss' : SiteMap
  ss' = occupy vacated dsite o

  m' : Machine
  m' = machine (heap m) (objects m) ss'

  held : Holds (sites m) ssite o
  held = strongAt-holds (sites m) ssite o at-src

  -- ⭐ The count does not move. One off, one on.
  ghost-same : ∀ p → ghostRC m' p ≡ ghostRC m p
  ghost-same p = go (sameObj o p) refl
    where
      go : (bb : Bool) → sameObj o p ≡ bb → ghostRC m' p ≡ ghostRC m p
      go true  e rewrite same→eq o p e =
        trans (occupy-same vacated dsite o)
              (sym (vacate-holder (sites m) ssite o at-src))
      go false e =
        trans (occupy-other vacated dsite o p e)
              (vacate-holder-other (sites m) ssite o p at-src e)

  counts : WFRC m'
  counts = record { counted-exact = ce
                  ; live-positive = lp
                  ; dead-unowned = du
                  ; no-stale-owner = ns
                  ; owned-storage-live = osl }
    where
      ce : ∀ p n → countOf m' p ≡ just (counted n) → lifeOf m' p ≡ just live →
           n ≡ ghostRC m' p
      ce p n cnt lif = trans (counted-exact (rc w) p n cnt lif) (sym (ghost-same p))

      lp : ∀ p → lifeOf m' p ≡ just live → IsCounted m' p → 0 < ghostRC m' p
      lp p lif cnt = subst (0 <_) (sym (ghost-same p)) (live-positive (rc w) p lif cnt)

      du : ∀ p → lifeOf m' p ≡ just dead → ghostRC m' p ≡ 0
      du p lif = trans (ghost-same p) (dead-unowned (rc w) p lif)

      ns : ∀ s p → Holds ss' s p →
           Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                    × (generation bb ≡ objGeneration p)
      ns s p holds-here      = no-stale-owner (rc w) ssite o held
      ns s p (holds-there h) =
        no-stale-owner (rc w) s p (holds-vacate (sites m) ssite s p h)

      osl : ∀ p → 0 < ghostRC m' p →
            Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                     × (liveness bb ≡ blockLive)
      osl p pos = owned-storage-live (rc w) p (subst (0 <_) (ghost-same p) pos)

module Move
  {t : ThreadId} {es : Env} {m : Machine} {src dst : Var} {o : ObjId}
  (look  : lookupVar es src ≡ just (bind o owned))
  (nodup : lookupVar (unbindVar es src) src ≡ nothing)
  (w     : WFES t es m)
  where

  private
    open module Core = MoveCore {t} {es} {m} {o} (siteOf t src) (siteOf t dst)
                                 (backed w src o look) w

  preserves : WFES t (bindVar (unbindVar es src) dst (bind o owned)) m'
  preserves = wfs counts bk'
    where

      -- Where the SSA premise is spent. Without it the source name could still
      -- be bound after the unbind -- a shadowed binding -- and would then be an
      -- owned name at a site the rule has just vacated.
      bk' : ∀ y q →
            lookupVar (bindVar (unbindVar es src) dst (bind o owned)) y
              ≡ just (bind q owned) →
            strongAt ss' (siteOf t y) ≡ just q
      bk' y q h = go (sameVar dst y) refl
        where
          rest : sameVar dst y ≡ false → lookupVar (unbindVar es src) y ≡ just (bind q owned)
          rest e = trans (sym (lookupVar-cons-false dst (bind o owned) (unbindVar es src) y e)) h

          inner : (bb : Bool) → sameVar src y ≡ bb →
                  lookupVar (unbindVar es src) y ≡ just (bind q owned) →
                  strongAt vacated (siteOf t y) ≡ just q
          inner true  e h' with trans (sym (subst (λ z → lookupVar (unbindVar es src) z ≡ nothing)
                                                  (sameVar-sound src y e) nodup)) h'
          ...                 | ()
          inner false e h' =
            trans (strongAt-vacate-other (sites m) (siteOf t src) (siteOf t y)
                     (trans (sameSite-siteOf t src y) e))
                  (backed w y q (trans (sym (unbindVar-other es src y e)) h'))

          go : (bb : Bool) → sameVar dst y ≡ bb → strongAt ss' (siteOf t y) ≡ just q
          go true e =
            trans (strongAt-cons-true (siteOf t dst) o vacated (siteOf t y)
                     (trans (sameSite-siteOf t dst y) e))
                  (cong just (bind-obj (just-inj
                     (trans (sym (lookupVar-cons-true dst (bind o owned)
                                    (unbindVar es src) y e)) h))))
          go false e =
            trans (strongAt-cons-false (siteOf t dst) o vacated (siteOf t y)
                     (trans (sameSite-siteOf t dst y) e))
                  (inner (sameVar src y) refl (rest e))

------------------------------------------------------------------------
-- A NAME's entity changes hands to a site no name owns.
--
-- Everything about the counter is `MoveCore`'s, unchanged: one site vacated,
-- one occupied. What is left is smaller than `move`'s, because no name is bound
-- -- the destination is not any name's site, so occupying it cannot disturb
-- what `backed` says about names.
--
-- `notName` is that condition, and both users discharge it with `refl`: a
-- destination headed by a different constructor than `local` gives
-- `sameSite _ _ ≡ false` by construction rather than by a lemma. A parameter
-- rather than a constructor test, because the two users -- `setField` into
-- `field′ p k` and `callOut` into `callee t c` -- differ in nothing else, and
-- writing the proof twice is how the second one drifts from the first.

module MoveToSite
  {t : ThreadId} {es : Env} {m : Machine} {src : Var} {o : ObjId}
  (dsite   : OwnerSite)
  (notName : ∀ y → sameSite dsite (siteOf t y) ≡ false)
  (look    : lookupVar es src ≡ just (bind o owned))
  (nodup   : lookupVar (unbindVar es src) src ≡ nothing)
  (w       : WFES t es m)
  where

  private
    open module Core = MoveCore {t} {es} {m} {o} (siteOf t src) dsite
                                 (backed w src o look) w

  preserves : WFES t (unbindVar es src) m'
  preserves = wfs counts bk'
    where
      bk' : ∀ y q → lookupVar (unbindVar es src) y ≡ just (bind q owned) →
            strongAt ss' (siteOf t y) ≡ just q
      bk' y q h =
        trans (strongAt-cons-false dsite o vacated (siteOf t y) (notName y))
              (inner (sameVar src y) refl)
        where
          -- The SSA premise, spent exactly as `move` spends it: without it the
          -- source could still be bound after the unbind and would be an owned
          -- name at the site the rule has just vacated.
          inner : (bb : Bool) → sameVar src y ≡ bb →
                  strongAt vacated (siteOf t y) ≡ just q
          inner true e
            with trans (sym (subst (λ z → lookupVar (unbindVar es src) z ≡ nothing)
                                   (sameVar-sound src y e) nodup)) h
          ...  | ()
          inner false e =
            trans (strongAt-vacate-other (sites m) (siteOf t src) (siteOf t y)
                     (trans (sameSite-siteOf t src y) e))
                  (backed w y q (trans (sym (unbindVar-other es src y e)) h))

------------------------------------------------------------------------
-- A site no name owns hands its entity TO a name.
--
-- The mirror of `MoveToSite`, and the reason `MoveCore` now takes a source
-- site: there is no source name here to be `backed` by one. `callIn` is the
-- only user, and the shape is `alloc`'s binding half over a vacated map rather
-- than over `sites m` -- which costs exactly one extra step, because vacating a
-- site no name owns cannot disturb what `backed` says about names.

module MoveFromSite
  {t : ThreadId} {es : Env} {m : Machine} {dst : Var} {o : ObjId}
  (ssite   : OwnerSite)
  (notName : ∀ y → sameSite ssite (siteOf t y) ≡ false)
  (at-src  : strongAt (sites m) ssite ≡ just o)
  (w       : WFES t es m)
  where

  private
    open module Core = MoveCore {t} {es} {m} {o} ssite (siteOf t dst) at-src w

  preserves : WFES t (bindVar es dst (bind o owned)) m'
  preserves = wfs counts bk'
    where
      bk' : ∀ y q →
            lookupVar (bindVar es dst (bind o owned)) y ≡ just (bind q owned) →
            strongAt ss' (siteOf t y) ≡ just q
      bk' y q h = go (sameVar dst y) refl
        where
          go : (bb : Bool) → sameVar dst y ≡ bb → strongAt ss' (siteOf t y) ≡ just q
          go true e =
            trans (strongAt-cons-true (siteOf t dst) o vacated (siteOf t y)
                     (trans (sameSite-siteOf t dst y) e))
                  (cong just (bind-obj (just-inj
                     (trans (sym (lookupVar-cons-true dst (bind o owned) es y e)) h))))
          go false e =
            trans (strongAt-cons-false (siteOf t dst) o vacated (siteOf t y)
                     (trans (sameSite-siteOf t dst y) e))
            (trans (strongAt-vacate-other (sites m) ssite (siteOf t y) (notName y))
                   (backed w y q
                      (trans (sym (lookupVar-cons-false dst (bind o owned) es y e))
                             h)))

------------------------------------------------------------------------
-- `drop` = py.decref. Name, site and counter all go together.
--
-- The case that matters is the one that reaches zero: the cell moves to
-- `finalizing`, NOT to `dead`, and not to `live` either. That is why the
-- reached-zero branch discharges `counted-exact` by contradiction with the life
-- premise rather than by arithmetic -- at zero the object is no longer live, so
-- the field says nothing about it, which is exactly the licence a finalizer
-- needs to run while the counter reads zero.

module Drop
  {t : ThreadId} {es : Env} {m : Machine} {x : Var} {o : ObjId} {c : ObjCell}
  (look  : lookupVar es x ≡ just (bind o owned))
  (tbl   : lookupObj (objects m) o ≡ just c)
  (alive : life c ≡ live)
  (nodup : lookupVar (unbindVar es x) x ≡ nothing)
  (w     : WFES t es m)
  where

  private
    ts' : ObjTable
    ts' = stepDownAt (objects m) o

    ss' : SiteMap
    ss' = vacate (sites m) (siteOf t x)

    m' : Machine
    m' = machine (heap m) ts' ss'

    at-x : strongAt (sites m) (siteOf t x) ≡ just o
    at-x = backed w x o look

    hit-cell : lookupObj ts' o ≡ just (stepDownCell c)
    hit-cell = lookupObj-update-same (objects m) o stepDownCell c tbl

    count-hit : countOf m' o ≡ just (bumpDown (count c))
    count-hit = count-from ts' o (stepDownCell c) hit-cell

    life-hit : lifeOf m' o ≡ just (lifeAfterDown (bumpDown (count c)) (life c))
    life-hit = life-from ts' o (stepDownCell c) hit-cell

    tbl-miss : ∀ p → sameObj o p ≡ false → lookupObj ts' p ≡ lookupObj (objects m) p
    tbl-miss p e = lookupObj-update-other (objects m) o p stepDownCell e

    count-miss : ∀ p → sameObj o p ≡ false → countOf m' p ≡ countOf m p
    count-miss p e = count-eq ts' (objects m) p (tbl-miss p e)

    life-miss : ∀ p → sameObj o p ≡ false → lifeOf m' p ≡ lifeOf m p
    life-miss p e = life-eq ts' (objects m) p (tbl-miss p e)

    -- ⭐ The count really did go down by one, because the site really was held.
    ghost-hit : ghostRC m o ≡ suc (ghostRC m' o)
    ghost-hit = vacate-holder (sites m) (siteOf t x) o at-x

    ghost-miss : ∀ p → sameObj o p ≡ false → ghostRC m' p ≡ ghostRC m p
    ghost-miss p e = vacate-holder-other (sites m) (siteOf t x) o p at-x e

    prior : ∀ k → count c ≡ counted (suc k) → suc k ≡ ghostRC m o
    prior k e = counted-exact (rc w) o (suc k)
                  (trans (count-from (objects m) o c tbl) (cong just e))
                  (trans (life-from (objects m) o c tbl) (cong just alive))

    ce-hit : ∀ n → countOf m' o ≡ just (counted n) → lifeOf m' o ≡ just live →
             n ≡ ghostRC m' o
    ce-hit n cnt lif = go (count c) refl
      where
        go : (r : RuntimeCount) → count c ≡ r → n ≡ ghostRC m' o
        go (counted zero) e
          with trans (sym lif)
                     (trans life-hit (cong (λ r → just (lifeAfterDown (bumpDown r) (life c))) e))
        ...  | ()
        go (counted (suc k)) e =
          trans (counted-inj (just-inj
                  (trans (sym cnt) (trans count-hit (cong (λ r → just (bumpDown r)) e)))))
                (suc-inj (trans (prior k e) ghost-hit))
        go immortal e
          with trans (sym cnt) (trans count-hit (cong (λ r → just (bumpDown r)) e))
        ...  | ()

    -- Positive because the surviving counter says so, and the counter cannot
    -- read zero while the cell still says `live`: `lifeAfterDown` sends exactly
    -- the zero case to `finalizing`.
    lp-hit : ∀ n → countOf m' o ≡ just (counted n) → lifeOf m' o ≡ just live →
             0 < ghostRC m' o
    lp-hit n cnt lif = go (bumpDown (count c)) refl
      where
        go : (r : RuntimeCount) → bumpDown (count c) ≡ r → 0 < ghostRC m' o
        go (counted zero) e
          with trans (sym lif) (trans life-hit (cong (λ r → just (lifeAfterDown r (life c))) e))
        ...  | ()
        go (counted (suc k)) e =
          subst (0 <_) (ce-hit n cnt lif)
            (subst (0 <_)
              (sym (counted-inj (just-inj (trans (sym cnt) (trans count-hit (cong just e))))))
              (s≤s z≤n))
        go immortal e with trans (sym cnt) (trans count-hit (cong just e))
        ...              | ()

    -- ⭐ `drop` cannot produce a dead object. Reaching zero gives `finalizing`,
    -- and every other outcome is the cell's own life, which the premise says is
    -- `live`. So the field holds at `o` vacuously -- which is the model's way of
    -- recording that a decref is not a free.
    du-hit : lifeOf m' o ≡ just dead → ghostRC m' o ≡ 0
    du-hit lif = go (bumpDown (count c)) refl
      where
        go : (r : RuntimeCount) → bumpDown (count c) ≡ r → ghostRC m' o ≡ 0
        go (counted zero) e
          with trans (sym lif) (trans life-hit (cong (λ r → just (lifeAfterDown r (life c))) e))
        ...  | ()
        go (counted (suc k)) e
          with trans (sym lif)
                     (trans life-hit
                       (trans (cong (λ r → just (lifeAfterDown r (life c))) e)
                              (cong just alive)))
        ...  | ()
        go immortal e
          with trans (sym lif)
                     (trans life-hit
                       (trans (cong (λ r → just (lifeAfterDown r (life c))) e)
                              (cong just alive)))
        ...  | ()

  preserves : WFES t (unbindVar es x) m'
  preserves = wfs (record { counted-exact = ce
                          ; live-positive = lp
                          ; dead-unowned = du
                          ; no-stale-owner = ns
                          ; owned-storage-live = osl }) bk'
    where
      ce : ∀ p n → countOf m' p ≡ just (counted n) → lifeOf m' p ≡ just live →
           n ≡ ghostRC m' p
      ce p n cnt lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → n ≡ ghostRC m' p
          go true  e rewrite same→eq o p e = ce-hit n cnt lif
          go false e = trans (counted-exact (rc w) p n
                               (trans (sym (count-miss p e)) cnt)
                               (trans (sym (life-miss p e)) lif))
                             (sym (ghost-miss p e))

      lp : ∀ p → lifeOf m' p ≡ just live → IsCounted m' p → 0 < ghostRC m' p
      lp p lif (n , cnt) = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → 0 < ghostRC m' p
          go true  e rewrite same→eq o p e = lp-hit n cnt lif
          go false e = subst (0 <_) (sym (ghost-miss p e))
                         (live-positive (rc w) p (trans (sym (life-miss p e)) lif)
                           (n , trans (sym (count-miss p e)) cnt))

      du : ∀ p → lifeOf m' p ≡ just dead → ghostRC m' p ≡ 0
      du p lif = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb → ghostRC m' p ≡ 0
          go true  e rewrite same→eq o p e = du-hit lif
          go false e = trans (ghost-miss p e)
                             (dead-unowned (rc w) p (trans (sym (life-miss p e)) lif))

      ns : ∀ s p → Holds ss' s p →
           Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                    × (generation bb ≡ objGeneration p)
      ns s p h = no-stale-owner (rc w) s p (holds-vacate (sites m) (siteOf t x) s p h)

      osl : ∀ p → 0 < ghostRC m' p →
            Σ _ λ bb → (lookupBlock (heap m) (objAllocation p) ≡ just bb)
                     × (liveness bb ≡ blockLive)
      osl p pos = go (sameObj o p) refl
        where
          go : (bb : Bool) → sameObj o p ≡ bb →
               Σ _ λ bb′ → (lookupBlock (heap m) (objAllocation p) ≡ just bb′)
                         × (liveness bb′ ≡ blockLive)
          go true  e rewrite same→eq o p e =
            owned-storage-live (rc w) o (subst (0 <_) (sym ghost-hit) (s≤s z≤n))
          go false e = owned-storage-live (rc w) p (subst (0 <_) (ghost-miss p e) pos)

      bk' : ∀ y q → lookupVar (unbindVar es x) y ≡ just (bind q owned) →
            strongAt ss' (siteOf t y) ≡ just q
      bk' y q h = go (sameVar x y) refl
        where
          go : (bb : Bool) → sameVar x y ≡ bb → strongAt ss' (siteOf t y) ≡ just q
          go true e with trans (sym (subst (λ z → lookupVar (unbindVar es x) z ≡ nothing)
                                           (sameVar-sound x y e) nodup)) h
          ...          | ()
          go false e =
            trans (strongAt-vacate-other (sites m) (siteOf t x) (siteOf t y)
                     (trans (sameSite-siteOf t x y) e))
                  (backed w y q (trans (sym (unbindVar-other es x y e)) h))

------------------------------------------------------------------------
-- Block arguments.
--
-- Stated over an `Env × SiteMap` against a FIXED heap and object table rather
-- than over a `Machine`, because the intermediate states of `moveArgs` are not
-- machines -- and inventing one per argument would have meant inventing an
-- object table per argument, which is precisely what "a block argument touches
-- no counter" denies.

moveOne-preserves :
  ∀ (t : ThreadId) (m : Machine) (es : Env) (ss : SiteMap) (p a : Var)
    (es' : Env) (ss' : SiteMap) →
  moveOne t (es , ss) p a ≡ just (es' , ss') →
  WFES t es (afterArgs m ss) → WFES t es' (afterArgs m ss')
moveOne-preserves t m es ss p a es' ss' eq w = outer (lookupVar es a) refl
  where
    tail-of : Binding → Maybe (Env × SiteMap)
    tail-of b = maybe′ (λ _ → nothing)
                       (just (bindVar (unbindVar es a) p b , relocate t ss a p b))
                       (lookupVar (unbindVar es a) a)

    -- A borrowed argument moves a NAME and no site. `unbindVar` only shrinks
    -- the environment, so every surviving owned name keeps the site it had.
    borrowed-case : ∀ o v →
      lookupVar (unbindVar es a) a ≡ nothing →
      WFES t (bindVar (unbindVar es a) p (bind o (borrowed v))) (afterArgs m ss)
    borrowed-case o v nodup = wfs (rc w) bk'
      where
        bk' : ∀ y q →
              lookupVar (bindVar (unbindVar es a) p (bind o (borrowed v))) y
                ≡ just (bind q owned) →
              strongAt ss (siteOf t y) ≡ just q
        bk' y q h = go (sameVar p y) refl
          where
            inner : (bb : Bool) → sameVar a y ≡ bb →
                    lookupVar (unbindVar es a) y ≡ just (bind q owned) →
                    strongAt ss (siteOf t y) ≡ just q
            inner true e h'
              with trans (sym (subst (λ z → lookupVar (unbindVar es a) z ≡ nothing)
                                     (sameVar-sound a y e) nodup)) h'
            ...  | ()
            inner false e h' = backed w y q (trans (sym (unbindVar-other es a y e)) h')

            go : (bb : Bool) → sameVar p y ≡ bb → strongAt ss (siteOf t y) ≡ just q
            go true e
              with trans (sym (lookupVar-cons-true p (bind o (borrowed v))
                                 (unbindVar es a) y e)) h
            ...  | ()
            go false e =
              inner (sameVar a y) refl
                (trans (sym (lookupVar-cons-false p (bind o (borrowed v))
                               (unbindVar es a) y e)) h)

    result : ∀ (b : Binding) →
             lookupVar es a ≡ just b →
             lookupVar (unbindVar es a) a ≡ nothing →
             WFES t (bindVar (unbindVar es a) p b) (afterArgs m (relocate t ss a p b))
    result (bind o owned)        look nodup = Move.preserves look nodup w
    result (bind o (borrowed v)) look nodup = borrowed-case o v nodup

    -- Two nested case analyses on the two lookups `moveOne` performs. Written
    -- as functions taking the equation rather than with `with`, for the reason
    -- recorded throughout this development: a `with` on either lookup would not
    -- reduce the occurrence sitting inside `eq`.
    middle : ∀ (b : Binding) → lookupVar es a ≡ just b →
             (r : Maybe Binding) → lookupVar (unbindVar es a) a ≡ r →
             maybe′ (λ _ → nothing)
                    (just (bindVar (unbindVar es a) p b , relocate t ss a p b))
                    r ≡ just (es' , ss') →
             WFES t es' (afterArgs m ss')
    middle b look (just _) e h with h
    ...                          | ()
    middle b look nothing  e h =
      subst₂ (λ E S → WFES t E (afterArgs m S))
             (cong proj₁ (just-inj h)) (cong proj₂ (just-inj h))
             (result b look e)

    outer : (r : Maybe Binding) → lookupVar es a ≡ r → WFES t es' (afterArgs m ss')
    outer nothing e
      with trans (sym (cong (maybe′ tail-of nothing) e)) eq
    ...  | ()
    outer (just b) e =
      middle b e (lookupVar (unbindVar es a) a) refl
             (trans (sym (cong (maybe′ tail-of nothing) e)) eq)

moveArgs-preserves :
  ∀ (t : ThreadId) (m : Machine) (es : Env) (ss : SiteMap) (ps as : List Var)
    (es' : Env) (ss' : SiteMap) →
  moveArgs t (es , ss) ps as ≡ just (es' , ss') →
  WFES t es (afterArgs m ss) → WFES t es' (afterArgs m ss')
moveArgs-preserves t m es ss [] [] es' ss' eq w =
  subst₂ (λ E S → WFES t E (afterArgs m S))
         (cong proj₁ (just-inj eq)) (cong proj₂ (just-inj eq)) w
moveArgs-preserves t m es ss (p ∷ ps) (a ∷ as) es' ss' eq w =
  go (moveOne t (es , ss) p a) refl
  where
    go : (r : Maybe (Env × SiteMap)) → moveOne t (es , ss) p a ≡ r →
         WFES t es' (afterArgs m ss')
    go nothing e
      with trans (sym (cong (maybe′ (λ st' → moveArgs t st' ps as) nothing) e)) eq
    ...  | ()
    go (just (es₁ , ss₁)) e =
      moveArgs-preserves t m es₁ ss₁ ps as es' ss'
        (trans (sym (cong (maybe′ (λ st' → moveArgs t st' ps as) nothing) e)) eq)
        (moveOne-preserves t m es ss p a es₁ ss₁ e w)
moveArgs-preserves t m es ss []       (_ ∷ _) es' ss' eq w with eq
...                                                          | ()
moveArgs-preserves t m es ss (_ ∷ _)  []      es' ss' eq w with eq
...                                                          | ()

------------------------------------------------------------------------
-- ⭐ THE THEOREMS.

instr-preserves-WF : ∀ {f s u} → f ⊢ s —→ᵢ u → WF s → WF u
instr-preserves-WF (step-alloc fresh unowned blk gen alv) w =
  Alloc.preserves fresh unowned blk gen alv w
instr-preserves-WF (step-init look fresh alone)        w = Init.preserves look fresh alone w
instr-preserves-WF (step-move look nodup)              w = Move.preserves look nodup w
instr-preserves-WF (step-dup look tbl alive)           w = Dup.preserves look tbl alive w
instr-preserves-WF (step-drop look tbl alive nodup)    w = Drop.preserves look tbl alive nodup w
instr-preserves-WF (step-borrow look)                  w = borrowed-bind-preserves w
instr-preserves-WF (step-set-field _ look nodup _)     w =
  MoveToSite.preserves _ (λ _ → refl) look nodup w
-- The transfer half of a call boundary, and the whole of its proof: a move to a
-- site no name owns, which `setField` already was.
instr-preserves-WF (step-call-out look nodup)         w =
  MoveToSite.preserves _ (λ _ → refl) look nodup w
-- And the receiving half, which is the same move read backwards.
instr-preserves-WF (step-call-in at-src)              w =
  MoveFromSite.preserves _ (λ _ → refl) at-src w
instr-preserves-WF (step-get-field _ _)                w = borrowed-bind-preserves w

-- Every terminator is the same operation on the state -- move the operands into
-- the parameters -- so all five constructors have one proof.
term-preserves-WF : ∀ {f s u} → f ⊢ s —→ₜ u → WF s → WF u
term-preserves-WF {s = pstate t bid [] es m}
  (step-br {args = args} {nxt = nxt} _ _ _ mv) w =
  moveArgs-preserves t m es (sites m) (params nxt) args _ _ mv w
term-preserves-WF {s = pstate t bid [] es m}
  (step-cond-then {a₁ = a₁} {nxt = nxt} _ _ _ mv) w =
  moveArgs-preserves t m es (sites m) (params nxt) a₁ _ _ mv w
term-preserves-WF {s = pstate t bid [] es m}
  (step-cond-else {a₂ = a₂} {nxt = nxt} _ _ _ mv) w =
  moveArgs-preserves t m es (sites m) (params nxt) a₂ _ _ mv w
term-preserves-WF {s = pstate t bid [] es m}
  (step-invoke-normal {a = a} {nxt = nxt} _ _ _ mv) w =
  moveArgs-preserves t m es (sites m) (params nxt) a _ _ mv w
term-preserves-WF {s = pstate t bid [] es m}
  (step-invoke-throw {pa = pa} {nxt = nxt} _ _ _ mv) w =
  moveArgs-preserves t m es (sites m) (params nxt) pa _ _ mv w

step-preserves-WF : ∀ {f s u} → f ⊢ s —→ u → WF s → WF u
step-preserves-WF (by-instr st) w = instr-preserves-WF st w
step-preserves-WF (by-term  st) w = term-preserves-WF st w

-- The refcount invariant survives one step. This is the statement the whole
-- development is for: `WFRC` constrains the states a program can be IN, not
-- merely the ones someone wrote down.
step-preserves : ∀ {f s u} → f ⊢ s —→ u → WF s → WFRC (mach u)
step-preserves st w = rc (step-preserves-WF st w)

reachable-preserves-WF : ∀ {f s u} → f ⊢ s —→* u → WF s → WF u
reachable-preserves-WF done        w = w
reachable-preserves-WF (more p ps) w = reachable-preserves-WF ps (step-preserves-WF p w)

-- ⭐ And over any number of steps. Nothing a program can reach from a
-- well-formed state violates the reference-count invariant.
reachable-preserves-WFRC : ∀ {f s u} → f ⊢ s —→* u → WF s → WFRC (mach u)
reachable-preserves-WFRC r w = rc (reachable-preserves-WF r w)

------------------------------------------------------------------------
-- What a terminator costs.

term-keeps-the-heap : ∀ {f s u} → f ⊢ s —→ₜ u → heap (mach u) ≡ heap (mach s)
term-keeps-the-heap (step-br _ _ _ _)            = refl
term-keeps-the-heap (step-cond-then _ _ _ _)     = refl
term-keeps-the-heap (step-cond-else _ _ _ _)     = refl
term-keeps-the-heap (step-invoke-normal _ _ _ _) = refl
term-keeps-the-heap (step-invoke-throw _ _ _ _)  = refl

-- ⭐ Passing block arguments costs ZERO runtime operations.
--
-- Not "the compiler currently emits none" but "no counter changes, so none is
-- required". This is the model's answer to the loop-carried retain/release pair
-- the compiler used to emit per iteration: the pair is not an optimisation the
-- backend may skip, it is an operation the semantics never asked for.
--
-- Stated over the whole object TABLE rather than over one object's counter,
-- because the weaker form is true of a rule that swaps two cells and the point
-- is that a terminator touches none of them.
block-arguments-are-free :
  ∀ {f s u} → f ⊢ s —→ₜ u → objects (mach u) ≡ objects (mach s)
block-arguments-are-free (step-br _ _ _ _)            = refl
block-arguments-are-free (step-cond-then _ _ _ _)     = refl
block-arguments-are-free (step-cond-else _ _ _ _)     = refl
block-arguments-are-free (step-invoke-normal _ _ _ _) = refl
block-arguments-are-free (step-invoke-throw _ _ _ _)  = refl

term-keeps-the-counters :
  ∀ {f s u} → f ⊢ s —→ₜ u → ∀ o → countOf (mach u) o ≡ countOf (mach s) o
term-keeps-the-counters st o =
  cong (λ ts → Data.Maybe.map count (lookupObj ts o)) (block-arguments-are-free st)
