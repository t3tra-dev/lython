{-# OPTIONS --safe #-}

-- Where an owner site's object meets its address.
--
-- The two layers already agreed on identity: `ObjId` is (allocation,
-- generation), the same pair the byte heap uses and the same pair an
-- `AlignedPointer` carries. What was missing is the step between them, and it
-- is the step the compiler takes constantly:
--
--     a slot holds an object; physically it holds that object's ADDRESS; a read
--     turns the address back into something usable.
--
-- `Proof.MemRef.Dialect` supplies the two halves -- `extractAlignedPointerAsIndex`
-- out, `descFromAlignedPointer` back, the second premised on the allocation
-- being live at a matching generation. Nothing said those premises ever hold,
-- so the rule was there and constrained nothing: the same shape as the record
-- it was written to repair, which named an obligation and never stated it.
--
-- Here they are discharged from `WFRC`, so a reference parked as an address is
-- not a weaker thing than a reference held as a value. For as long as a site
-- holds it, the address recovers, and what it recovers is the object the site
-- said was there.
--
-- Why this matters to the compiler and not only to the development: a `memref`
-- cannot have a pointer element type, so every reference a boxed Lython object
-- owns is an address in an `i64` and every read of one is an `inttoptr`. That
-- cannot be removed -- and these theorems are what make it not a hole.

open import Proof.Memory.Element using (ElemSig)

module Proof.RC.Address (Sig : ElemSig) where

open ElemSig Sig

open import Data.Empty using (⊥-elim)
open import Data.Maybe using (Maybe; just; nothing)
open import Data.Maybe.Properties using (just-injective)
open import Data.Nat using (ℕ; _≟_)
open import Data.Product using (Σ; _×_; _,_)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans)
open import Relation.Nullary using (yes; no)

open import Proof.Prelude using (Result; err; ok)
open import Proof.Memory.Heap using (Heap; Block; lookupBlock; generation;
  liveness; space)
  renaming (live to blockLive; dead to blockDead)
open import Proof.Memory.Descriptor Sig using (Desc; desc; allocation)
  renaming (generation to descGeneration)
open import Proof.MemRef.Dialect Sig using (AlignedPointer; alignedPtr;
  ptrAllocation; ptrGeneration; extractAlignedPointerAsIndex;
  descFromAlignedPointer)
open import Proof.RC.Object using (ObjId; obj; objAllocation; objGeneration)
open import Proof.RC.OwnerSite using (OwnerSite; Holds; holds-positive)
open import Proof.RC.Machine Sig using (Machine; heap; sites)
open import Proof.RC.Invariant Sig using (WFRC; no-stale-owner;
  owned-storage-live)

------------------------------------------------------------------------
-- The correspondence, both ways.

-- Which object a descriptor is over. Neither layer could have defined this on
-- its own: the memory layer has no notion of an object, and the refcount layer
-- has no descriptors.
objOf : ∀ {r} → Desc r → ObjId
objOf d = obj (allocation d) (descGeneration d)

-- An object's address. Byte offset 0, because this is the object's own storage
-- rather than a view into it -- which is what a field slot holds.
addressOf : ObjId → AlignedPointer
addressOf o = alignedPtr (objAllocation o) (objGeneration o) 0

-- Out: taking a descriptor's address keeps the object it was over. Trivial to
-- prove and not trivial to say -- it is why `extract_aligned_pointer_as_index`
-- is modelled as yielding an identity rather than a number.
extract-keeps-object :
  ∀ {r} (d : Desc r) →
  obj (ptrAllocation (extractAlignedPointerAsIndex d))
      (ptrGeneration (extractAlignedPointerAsIndex d)) ≡ objOf d
extract-keeps-object d = refl

------------------------------------------------------------------------
-- The join.

private
  -- The three facts `descFromAlignedPointer` asks for, assembled into its
  -- success. Separate from the theorem below so that the theorem is about
  -- WFRC and this is about the function.
  recoverFrom :
    ∀ (h : Heap) (o : ObjId) (τ : ElemTy) (count : ℕ) (b : Block) →
    lookupBlock h (objAllocation o) ≡ just b →
    generation b ≡ objGeneration o →
    liveness b ≡ blockLive →
    Σ (Desc 1) λ d →
      (descFromAlignedPointer h (addressOf o) τ count ≡ ok d) × (objOf d ≡ o)
  recoverFrom h o τ count b found genMatch alive rewrite found
    with generation b ≟ objGeneration o | liveness b | alive
  ... | no ¬same | _         | _  = ⊥-elim (¬same genMatch)
  ... | yes _    | blockDead | ()
  ... | yes _    | blockLive | _  = _ , refl , refl

-- THE ONE. While a site holds an object, that object's address recovers, and
-- what it recovers is that object.
--
-- Both premises come from the invariant and neither is an assumption about the
-- compiler: `no-stale-owner` gives the block and the generation match,
-- `owned-storage-live` gives liveness, reached from the site through
-- `holds-positive`. So `use-after-free` and `stale-generation` -- the two ways
-- the trip back can fail -- are unreachable for a held reference, which is the
-- property a box word's `inttoptr` needs and previously had no way to claim.
site-address-recovers :
  ∀ (m : Machine) (s : OwnerSite) (o : ObjId) (τ : ElemTy) (count : ℕ) →
  WFRC m → Holds (sites m) s o →
  Σ (Desc 1) λ d →
    (descFromAlignedPointer (heap m) (addressOf o) τ count ≡ ok d)
      × (objOf d ≡ o)
site-address-recovers m s o τ count wf held
  with no-stale-owner wf s o held
     | owned-storage-live wf o (holds-positive (sites m) s o held)
... | (b , found , genMatch) | (b' , found' , alive) =
  recoverFrom (heap m) o τ count b found genMatch
    (subst-live (just-injective (trans (sym found) found')))
  where
    -- `lookupBlock` is a function, so the block the generation field found and
    -- the block the liveness field found are the same one.
    subst-live : b ≡ b' → liveness b ≡ blockLive
    subst-live refl = alive
