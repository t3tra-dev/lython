{-# OPTIONS --safe #-}

-- The faults this model distinguishes.
--
-- MLIR calls most of these "undefined behaviour". This development does NOT
-- model them as "anything may happen": an undefined operation gets a specific
-- constructor here, and the safety theorem is then the statement that a
-- verified program never reaches one. Collapsing them into a single `bad`
-- would make that theorem true and useless -- it has to be possible to say
-- WHICH guarantee a program would have broken.

module Proof.Memory.Fault where

data MemoryFault : Set where
  -- The index resolved outside the allocation.
  out-of-bounds        : MemoryFault
  -- The allocation exists but is no longer live.
  use-after-free       : MemoryFault
  -- dealloc of an allocation that is already not live.
  double-free          : MemoryFault
  -- dealloc through a descriptor that is not the allocation's root.
  invalid-free         : MemoryFault
  -- The descriptor's generation does not match the block's: the allocation id
  -- was reused, or realloc moved the storage. Distinct from use-after-free
  -- because the block IS live -- it is a different block.
  stale-generation     : MemoryFault
  -- Every byte was in bounds, and at least one had never been written.
  uninitialized-read   : MemoryFault
  misaligned-access    : MemoryFault
  invalid-memory-space : MemoryFault
  -- The descriptor's runtime metadata contradicts its static type.
  descriptor-overflow  : MemoryFault
  -- No allocation with that id exists at all.
  no-such-allocation   : MemoryFault

-- Why these two are separate: the first says a descriptor's own indices alias
-- each other, which MLIR's memref type forbids outright; the second says two
-- DIFFERENT descriptors alias, which is legal and is exactly what subview and
-- view produce. A single "aliasing" fault would reject the legal case.
data LayoutFault : Set where
  self-aliasing-layout : LayoutFault
  negative-byte-start  : LayoutFault
