{-# OPTIONS --safe #-}

-- Where an owning reference can live, and the count that follows from it.
--
-- The central move of the whole refcount layer:
--
--   the reference count is not a number the runtime maintains and the proof
--   trusts. It is the NUMBER OF OWNER SITES holding the object, computed from
--   ghost state -- and `counted-exact` in Proof.RC.Invariant is the obligation
--   that the runtime's counter implements it.
--
-- Stated the other way round -- "the count is whatever the counter says" --
-- there is nothing to violate, and no incref/decref placement can be wrong.

module Proof.RC.OwnerSite where

open import Data.Bool using (Bool; true; false; if_then_else_; _∧_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.List using (List; []; _∷_; length; filter)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Nat using (ℕ; zero; suc; _≟_; _+_; _<_; s≤s; z≤n)
open import Data.Nat.Base using (_≡ᵇ_)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality
  using (_≡_; _≢_; refl; sym; trans; cong; cong₂)
open import Relation.Nullary using (Dec; yes; no; ¬_)

open import Proof.RC.Object using (ObjId; obj; _≟-obj_; sameObj; sameObj-refl;
  sameObj-sound; ≡ᵇ-refl; ≡ᵇ-sound)

ThreadId LocalSlot FieldId GlobalId QueueId Ticket TempId CallId : Set
ThreadId  = ℕ
LocalSlot = ℕ
FieldId   = ℕ
GlobalId  = ℕ
QueueId   = ℕ
Ticket    = ℕ
TempId    = ℕ
CallId    = ℕ

-- Every place an owning reference can be. The list is not decoration: an owner
-- site the model forgets is a reference the count does not include, and the
-- invariant would then be provable about a machine that leaks.
--
-- `temp` is the one that is easy to leave out and is exactly where this
-- compiler's defects have lived: the transient reference held while a field is
-- being updated, or while a value is in flight into a container literal.
data OwnerSite : Set where
  local  : ThreadId → LocalSlot → OwnerSite
  field′ : ObjId → FieldId → OwnerSite
  global : GlobalId → OwnerSite
  queue  : QueueId → Ticket → OwnerSite
  temp   : ThreadId → TempId → OwnerSite
  -- ⭐ A reference held by an activation this trace does not step through.
  --
  -- Without it the model has no call, and that is not a small absence: six of
  -- the compiler's sixteen ownership attributes describe the call boundary and
  -- not one of them could be stated. The obstruction was exactly what this
  -- list's opening sentence warns about. An object returned +1 is counted by
  -- the callee and held at no site the model has, so `counted-exact` was
  -- violated BEFORE the caller's rule ran -- and no premise on that rule can
  -- repair a pre-state. The reference was not miscounted; it was in a place the
  -- list forgot.
  --
  -- With the site both directions are ordinary moves and the counter never
  -- moves with them: a transferred argument goes local -> callee. That is
  -- `setField`'s lesson again -- a store was expressible only because `field′`
  -- already named its destination.
  --
  -- Why NOT reuse `temp`, which would have cost no constructor: `temp` is the
  -- CALLER's own in-flight reference, and the container-literal defects it
  -- exists for are found by asking which temp is still occupied. Sharing it
  -- would make an unreturned call and a dropped container write one report.
  callee : ThreadId → CallId → OwnerSite

_≟-site_ : (s t : OwnerSite) → Dec (s ≡ t)
local t s   ≟-site local t' s'  with t ≟ t' | s ≟ s'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
field′ o f  ≟-site field′ o' f' with o ≟-obj o' | f ≟ f'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
global g    ≟-site global g'    with g ≟ g'
... | yes refl = yes refl
... | no ¬p    = no λ { refl → ¬p refl }
queue q k   ≟-site queue q' k'  with q ≟ q' | k ≟ k'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
temp t i    ≟-site temp t' i'   with t ≟ t' | i ≟ i'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
local _ _   ≟-site field′ _ _ = no λ ()
local _ _   ≟-site global _   = no λ ()
local _ _   ≟-site queue _ _  = no λ ()
local _ _   ≟-site temp _ _   = no λ ()
field′ _ _  ≟-site local _ _  = no λ ()
field′ _ _  ≟-site global _   = no λ ()
field′ _ _  ≟-site queue _ _  = no λ ()
field′ _ _  ≟-site temp _ _   = no λ ()
global _    ≟-site local _ _  = no λ ()
global _    ≟-site field′ _ _ = no λ ()
global _    ≟-site queue _ _  = no λ ()
global _    ≟-site temp _ _   = no λ ()
queue _ _   ≟-site local _ _  = no λ ()
queue _ _   ≟-site field′ _ _ = no λ ()
queue _ _   ≟-site global _   = no λ ()
queue _ _   ≟-site temp _ _   = no λ ()
temp _ _    ≟-site local _ _  = no λ ()
temp _ _    ≟-site field′ _ _ = no λ ()
temp _ _    ≟-site global _   = no λ ()
temp _ _    ≟-site queue _ _  = no λ ()
callee t k  ≟-site callee t' k' with t ≟ t' | k ≟ k'
... | yes refl | yes refl = yes refl
... | no ¬p    | _        = no λ { refl → ¬p refl }
... | _        | no ¬q    = no λ { refl → ¬q refl }
local _ _   ≟-site callee _ _ = no λ ()
field′ _ _  ≟-site callee _ _ = no λ ()
global _    ≟-site callee _ _ = no λ ()
queue _ _   ≟-site callee _ _ = no λ ()
temp _ _    ≟-site callee _ _ = no λ ()
callee _ _  ≟-site local _ _  = no λ ()
callee _ _  ≟-site field′ _ _ = no λ ()
callee _ _  ≟-site global _   = no λ ()
callee _ _  ≟-site queue _ _  = no λ ()
callee _ _  ≟-site temp _ _   = no λ ()

------------------------------------------------------------------------
-- The site map: which object, if any, each occupied site holds.
--
-- An association list rather than a function, so that `logicalRC` can COUNT it.
-- As a function `OwnerSite → Maybe ObjId` the count would need the domain to be
-- finite and enumerable, which is extra structure carried for no gain.

SiteMap : Set
SiteMap = List (OwnerSite × ObjId)

-- Boolean, for the same reason the object table is: `with s ≟-site t` compiles
-- to an auxiliary function, and then `vacate` and `logicalRC` branch on terms a
-- goal cannot abstract together. The counting lemmas below are unprovable
-- otherwise, and they are what WFRC preservation rests on.
sameSite : OwnerSite → OwnerSite → Bool
sameSite (local t s)  (local t' s')  = (t ≡ᵇ t') ∧ (s ≡ᵇ s')
sameSite (field′ o f) (field′ o' f') = sameObj o o' ∧ (f ≡ᵇ f')
sameSite (global g)   (global g')    = g ≡ᵇ g'
sameSite (queue q k)  (queue q' k')  = (q ≡ᵇ q') ∧ (k ≡ᵇ k')
sameSite (temp t i)   (temp t' i')   = (t ≡ᵇ t') ∧ (i ≡ᵇ i')
sameSite (callee t k) (callee t' k') = (t ≡ᵇ t') ∧ (k ≡ᵇ k')
sameSite _            _              = false

sameSite-refl : ∀ s → sameSite s s ≡ true
sameSite-refl (local t s)  rewrite ≡ᵇ-refl t | ≡ᵇ-refl s = refl
sameSite-refl (field′ o f) rewrite sameObj-refl o | ≡ᵇ-refl f = refl
sameSite-refl (global g)   rewrite ≡ᵇ-refl g = refl
sameSite-refl (queue q k)  rewrite ≡ᵇ-refl q | ≡ᵇ-refl k = refl
sameSite-refl (temp t i)   rewrite ≡ᵇ-refl t | ≡ᵇ-refl i = refl
sameSite-refl (callee t k) rewrite ≡ᵇ-refl t | ≡ᵇ-refl k = refl

-- Reflection, as for `sameObj`. Needed as soon as a proof has to turn "the
-- lookup matched here" into "this really is that site".
sameSite-sound : ∀ s t → sameSite s t ≡ true → s ≡ t
sameSite-sound (local a x)  (local b y)  e = go a b x y e
  where go : ∀ a b x y → ((a ≡ᵇ b) ∧ (x ≡ᵇ y)) ≡ true → local a x ≡ local b y
        go a b x y e with a ≡ᵇ b | ≡ᵇ-sound a b
        ... | true | f with x ≡ᵇ y | ≡ᵇ-sound x y
        ...   | true | g = cong₂ local (f refl) (g refl)
sameSite-sound (field′ o f) (field′ p g) e = go o p f g e
  where go : ∀ o p f g → (sameObj o p ∧ (f ≡ᵇ g)) ≡ true → field′ o f ≡ field′ p g
        go o p f g e with sameObj o p | sameObj-sound o p
        ... | true | u with f ≡ᵇ g | ≡ᵇ-sound f g
        ...   | true | v = cong₂ field′ (u refl) (v refl)
sameSite-sound (global a)   (global b)   e = cong global (≡ᵇ-sound a b e)
sameSite-sound (queue q k)  (queue r l)  e = go q r k l e
  where go : ∀ q r k l → ((q ≡ᵇ r) ∧ (k ≡ᵇ l)) ≡ true → queue q k ≡ queue r l
        go q r k l e with q ≡ᵇ r | ≡ᵇ-sound q r
        ... | true | f with k ≡ᵇ l | ≡ᵇ-sound k l
        ...   | true | g = cong₂ queue (f refl) (g refl)
sameSite-sound (temp a i)   (temp b j)   e = go a b i j e
  where go : ∀ a b i j → ((a ≡ᵇ b) ∧ (i ≡ᵇ j)) ≡ true → temp a i ≡ temp b j
        go a b i j e with a ≡ᵇ b | ≡ᵇ-sound a b
        ... | true | f with i ≡ᵇ j | ≡ᵇ-sound i j
        ...   | true | g = cong₂ temp (f refl) (g refl)
sameSite-sound (callee a i) (callee b j) e = go a b i j e
  where go : ∀ a b i j → ((a ≡ᵇ b) ∧ (i ≡ᵇ j)) ≡ true → callee a i ≡ callee b j
        go a b i j e with a ≡ᵇ b | ≡ᵇ-sound a b
        ... | true | f with i ≡ᵇ j | ≡ᵇ-sound i j
        ...   | true | g = cong₂ callee (f refl) (g refl)

-- and the reverse direction, which is what turns a `false` into a disequality.
sameSite-complete : ∀ {s t} → s ≡ t → sameSite s t ≡ true
sameSite-complete {s} refl = sameSite-refl s

sameSite-false : ∀ {s t} → sameSite s t ≡ false → s ≢ t
sameSite-false ne eq with trans (sym (sameSite-complete eq)) ne
... | ()

sameSite-sym : ∀ s t → sameSite s t ≡ sameSite t s
sameSite-sym s t with sameSite s t in st | sameSite t s in ts
... | true  | true  = refl
... | false | false = refl
... | true  | false = ⊥-elim (bad (trans (sym (sameSite-complete
                        (sym (sameSite-sound s t st)))) ts))
  where bad : true ≡ false → ⊥
        bad ()
... | false | true  = ⊥-elim (bad (trans (sym (sameSite-complete
                        (sym (sameSite-sound t s ts)))) st))
  where bad : true ≡ false → ⊥
        bad ()

strongAt : SiteMap → OwnerSite → Maybe ObjId
strongAt []             _ = nothing
strongAt ((s , o) ∷ ss) t = if sameSite s t then just o else strongAt ss t

-- The two equations, carried explicitly, for the same reason as everywhere else
-- in this development: reduction under a `vacate` or an `occupy` creates a
-- scrutinee no with-abstraction saw.
strongAt-cons-true : ∀ s q (ss : SiteMap) u → sameSite s u ≡ true →
                     strongAt ((s , q) ∷ ss) u ≡ just q
strongAt-cons-true s q ss u e rewrite e = refl

strongAt-cons-false : ∀ s q (ss : SiteMap) u → sameSite s u ≡ false →
                      strongAt ((s , q) ∷ ss) u ≡ strongAt ss u
strongAt-cons-false s q ss u e rewrite e = refl

-- THE definition. The reference count of an object is how many owner sites hold
-- it -- an ordinary count over ghost state, with no runtime counter involved.
logicalRC : SiteMap → ObjId → ℕ
logicalRC []             _ = 0
logicalRC ((_ , p) ∷ ss) o = if sameObj p o then suc (logicalRC ss o) else logicalRC ss o

-- ⭐ The same count, restricted to the sites NO NAME OWNS.
--
-- `logicalRC` counts every owner site; a name holds some of them and something
-- else holds the rest. The coherence invariant needs the split, because a step
-- that moves a hold off a name and onto one of the others leaves the name side
-- reading as a loss.
--
-- This was `fieldRC`, over `field′` alone, because `setField` was the only step
-- that moved a hold that way. `callOut` is a second -- a transferred argument
-- goes to `callee` -- and a third term in the invariant for every such site
-- would be a family that grows with the site list. The complement of `local` is
-- the property actually wanted: a hold is a name's, or it is not.
--
-- Nothing else about the model changes: this is `logicalRC` with one extra
-- conjunct, over the same list, so its lemmas are the same lemmas.
isUnnamedSite : OwnerSite → Bool
isUnnamedSite (local _ _) = false
isUnnamedSite _           = true

unnamedRC : SiteMap → ObjId → ℕ
unnamedRC []             _ = 0
unnamedRC ((s , p) ∷ ss) o =
  if isUnnamedSite s ∧ sameObj p o then suc (unnamedRC ss o) else unnamedRC ss o

------------------------------------------------------------------------
-- The three site operations, and what each does to the count.

occupy : SiteMap → OwnerSite → ObjId → SiteMap
occupy ss s o = (s , o) ∷ ss

vacate : SiteMap → OwnerSite → SiteMap
vacate []             _ = []
vacate ((s , o) ∷ ss) t = if sameSite s t then ss else (s , o) ∷ vacate ss t

------------------------------------------------------------------------
-- Counting lemmas.

occupy-same : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
              logicalRC (occupy ss s o) o ≡ suc (logicalRC ss o)
occupy-same ss s o rewrite sameObj-refl o = refl

occupy-other : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) → sameObj o p ≡ false →
               logicalRC (occupy ss s o) p ≡ logicalRC ss p
-- Stated over the BOOLEAN rather than over the proposition, because that is
-- what `logicalRC` branches on. Callers turn a disequality into it with
-- `sameObj-sound` contraposed.
occupy-other ss s o p ne rewrite ne = refl
-- What the site operations do to the UNNAMED half.
--
-- Occupying or vacating a site a name owns cannot move it, and that is true by
-- reduction rather than by induction for `occupy` -- the conjunct is already
-- `false`. `vacate` needs the induction, because the entry it removes is
-- somewhere in the list.
unnamedRC-occupy-named : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) →
                         isUnnamedSite s ≡ false →
                         unnamedRC (occupy ss s o) p ≡ unnamedRC ss p
unnamedRC-occupy-named ss s o p nf rewrite nf = refl

-- The destination is quantified now rather than being `field′ q k`, and that
-- costs exactly one case split: `isUnnamedSite (field′ q k)` reduced to `true`
-- on its own, and a variable does not, so `b ∧ false` has to be taken apart by
-- hand.
unnamedRC-occupy-same : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                        isUnnamedSite s ≡ true →
                        unnamedRC (occupy ss s o) o ≡ suc (unnamedRC ss o)
unnamedRC-occupy-same ss s o un rewrite un | sameObj-refl o = refl

unnamedRC-occupy-other : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) →
                         sameObj o p ≡ false →
                         unnamedRC (occupy ss s o) p ≡ unnamedRC ss p
unnamedRC-occupy-other ss s o p ne with isUnnamedSite s
... | true  rewrite ne = refl
... | false = refl

-- `sameSite` is true only between sites of the same constructor, so a site that
-- matches a name's is a name's. Stated over the boolean the counts branch on,
-- like every other lemma here.
sameSite-unnamed : ∀ (s t : OwnerSite) → sameSite s t ≡ true →
                   isUnnamedSite s ≡ isUnnamedSite t
sameSite-unnamed (local _ _)  (local _ _)  _ = refl
sameSite-unnamed (field′ _ _) (field′ _ _) _ = refl
sameSite-unnamed (global _)   (global _)   _ = refl
sameSite-unnamed (queue _ _)  (queue _ _)  _ = refl
sameSite-unnamed (temp _ _)   (temp _ _)   _ = refl
sameSite-unnamed (callee _ _) (callee _ _) _ = refl

unnamedRC-vacate-named : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                         isUnnamedSite s ≡ false →
                         unnamedRC (vacate ss s) o ≡ unnamedRC ss o
unnamedRC-vacate-named []             s o nf = refl
unnamedRC-vacate-named ((t , p) ∷ ss) s o nf = go (sameSite t s) refl
  where
    go : (bb : Bool) → sameSite t s ≡ bb →
         unnamedRC (vacate ((t , p) ∷ ss) s) o ≡ unnamedRC ((t , p) ∷ ss) o
    go true  e rewrite e | trans (sameSite-unnamed t s e) nf = refl
    go false e rewrite e =
      cong (λ n → if isUnnamedSite t ∧ sameObj p o then suc n else n)
           (unnamedRC-vacate-named ss s o nf)

-- ⭐ Vacating a site that HOLDS the object drops the count by exactly one.
--
-- This is the lemma WFRC preservation for `drop` needs, and the reason the site
-- map had to move to a boolean test: `strongAt` and `vacate` branch on the same
-- decision, and with `_≟-site_` they branch on two different auxiliary
-- functions that no single `with` abstracts.
vacate-holder : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                strongAt ss s ≡ just o →
                logicalRC ss o ≡ suc (logicalRC (vacate ss s) o)
vacate-holder [] s o ()
vacate-holder ((t , p) ∷ ss) s o held with sameSite t s
... | true  = helper (just-inj held)
  where
    just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
    just-inj refl = refl
    helper : p ≡ o → logicalRC ((t , p) ∷ ss) o ≡ suc (logicalRC ss o)
    helper refl rewrite sameObj-refl p = refl
... | false with sameObj p o
...   | true  = cong suc (vacate-holder ss s o held)
...   | false = vacate-holder ss s o held

-- And vacating a site that does NOT hold the object leaves its count alone.
-- Without this, `move` -- which vacates one site and occupies another -- could
-- not be shown to preserve anything.
vacate-other : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
               strongAt ss s ≡ nothing →
               logicalRC (vacate ss s) o ≡ logicalRC ss o
vacate-other [] s o _ = refl
vacate-other ((t , p) ∷ ss) s o miss with sameSite t s
... | true  = ⊥-elim (bad miss)
  where
    bad : just p ≡ nothing → ⊥
    bad ()
... | false with sameObj p o
...   | true  = cong suc (vacate-other ss s o miss)
...   | false = vacate-other ss s o miss

-- ⭐ Vacating a site that holds `o` leaves EVERY OTHER object's count alone.
--
-- The companion of `vacate-holder`, and the one `move` and `drop` preservation
-- need. `vacate-other` is not it: that one is about vacating a site that holds
-- nothing, which is not the case here -- the site holds something, just not the
-- object being counted.
vacate-holder-other : ∀ (ss : SiteMap) (s : OwnerSite) (o p : ObjId) →
                      strongAt ss s ≡ just o → sameObj o p ≡ false →
                      logicalRC (vacate ss s) p ≡ logicalRC ss p
vacate-holder-other [] s o p () _
vacate-holder-other ((t , q) ∷ ss) s o p held ne with sameSite t s
... | true  = helper (just-inj held)
  where
    just-inj : ∀ {A : Set} {u v : A} → just u ≡ just v → u ≡ v
    just-inj refl = refl
    -- `q ≡ o` and `sameObj o p ≡ false`, so the head does not contribute to
    -- `p`'s count and dropping it changes nothing.
    helper : q ≡ o → logicalRC ss p ≡ logicalRC ((t , q) ∷ ss) p
    helper refl rewrite ne = refl
... | false with sameObj q p
...   | true  = cong suc (vacate-holder-other ss s o p held ne)
...   | false = vacate-holder-other ss s o p held ne

-- Vacating one site does not disturb what any OTHER site reports.
strongAt-vacate-other : ∀ (ss : SiteMap) (s t : OwnerSite) →
                        sameSite s t ≡ false →
                        strongAt (vacate ss s) t ≡ strongAt ss t
strongAt-vacate-other []             s t _  = refl
strongAt-vacate-other ((u , p) ∷ ss) s t ne with sameSite u s in us
... | true  rewrite sym (sameSite-sound u s us) | ne = refl
... | false with sameSite u t
...   | true  = refl
...   | false = strongAt-vacate-other ss s t ne

------------------------------------------------------------------------
-- ⭐ Holding a site, as MEMBERSHIP rather than as a lookup.
--
-- `strongAt` reports the FIRST entry at a site, and that is the right thing for
-- reading. It is the wrong thing for an invariant, because it is not preserved
-- by `vacate`: removing the first entry at a site exposes whatever was behind
-- it, and a property that only constrained first-entries says nothing about the
-- exposed one.
--
-- `WFRC.no-stale-owner` was stated over `strongAt` and was therefore not
-- preservable by `drop`. Over `Holds` it is, and it is also the property one
-- actually wants: every reference the site map records has live storage at a
-- matching generation, not merely the ones a lookup happens to reach.

data Holds : SiteMap → OwnerSite → ObjId → Set where
  holds-here  : ∀ {ss s o} → Holds ((s , o) ∷ ss) s o
  holds-there : ∀ {ss s o t p} → Holds ss s o → Holds ((t , p) ∷ ss) s o

-- What a lookup finds really is held.
strongAt-holds : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                 strongAt ss s ≡ just o → Holds ss s o
strongAt-holds []             s o ()
strongAt-holds ((u , p) ∷ ss) s o held with sameSite u s in us
... | true  = helper (sameSite-sound u s us) (just-inj held)
  where
    just-inj : ∀ {A : Set} {x y : A} → just x ≡ just y → x ≡ y
    just-inj refl = refl
    helper : u ≡ s → p ≡ o → Holds ((u , p) ∷ ss) s o
    helper refl refl = holds-here
... | false = holds-there (strongAt-holds ss s o held)

-- Vacating removes; it never invents. So everything still held was held before.
holds-vacate : ∀ (ss : SiteMap) (t s : OwnerSite) (o : ObjId) →
               Holds (vacate ss t) s o → Holds ss s o
holds-vacate []             t s o ()
holds-vacate ((u , p) ∷ ss) t s o h with sameSite u t
... | true  = holds-there h
holds-vacate ((u , p) ∷ ss) t s o holds-here      | false = holds-here
holds-vacate ((u , p) ∷ ss) t s o (holds-there h) | false =
  holds-there (holds-vacate ss t s o h)

-- Anything held contributes to its object's count, so the count is positive.
-- This is what lets `owned-storage-live` be reached from a site rather than
-- from an arithmetic fact about the count.
holds-positive : ∀ (ss : SiteMap) (s : OwnerSite) (o : ObjId) →
                 Holds ss s o → 0 < logicalRC ss o
holds-positive ((s , o) ∷ ss) s o holds-here rewrite sameObj-refl o = s≤s z≤n
holds-positive ((t , p) ∷ ss) s o (holds-there h) with sameObj p o
... | true  = s≤s z≤n
... | false = holds-positive ss s o h
