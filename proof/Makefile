# Typecheck the memory-safety kernel.
#
#   make            typecheck everything
#   make clean      drop the interface files
#   make redcheck   prove the checker is live (a deliberately false lemma must fail)
#
# Why the library file is DISCOVERED rather than committed or installed globally:
# agda-stdlib ships with Homebrew but is not registered in ~/.agda/libraries, and
# its path is machine-specific. Writing it into the repo would hard-code one
# machine; appending to ~/.agda/libraries would edit the user's global config for
# a project-local dependency. So the path is found here and written to a
# gitignored file that `--library-file` consumes.
#
# Set AGDA_STDLIB to override, e.g.
#   make AGDA_STDLIB=/path/to/standard-library.agda-lib

AGDA ?= agda
LIBFILE := .agda-libraries

# Ordered by how specific each guess is, so an explicit AGDA_STDLIB always wins.
STDLIB_CANDIDATES := \
  $(AGDA_STDLIB) \
  $(HOME)/.agda/standard-library.agda-lib \
  /opt/homebrew/share/agda/stdlib/standard-library.agda-lib \
  /usr/local/share/agda/stdlib/standard-library.agda-lib \
  /usr/share/agda-stdlib/standard-library.agda-lib

STDLIB := $(firstword $(foreach c,$(STDLIB_CANDIDATES),$(wildcard $(c))))

.PHONY: all clean redcheck stdlib-check

all: stdlib-check $(LIBFILE)
	$(AGDA) --library-file=$(LIBFILE) src/Everything.agda

# Refuse rather than run: with no stdlib, `agda` fails with a scope error that
# reads like a mistake in the proof rather than a missing dependency.
stdlib-check:
ifeq ($(STDLIB),)
	@echo "agda-stdlib not found. Looked in:" >&2
	@$(foreach c,$(STDLIB_CANDIDATES),echo "  $(c)" >&2;)
	@echo "Set AGDA_STDLIB=/path/to/standard-library.agda-lib" >&2
	@exit 2
else
	@true
endif

$(LIBFILE): Makefile
	@echo "$(STDLIB)" > $@
	@echo "$(CURDIR)/lython-proof.agda-lib" >> $@

# A checker that accepts everything would report this kernel as proved, so the
# checker itself needs checking. TWO sentinels, and both directions matter:
#
#   Positive -- a TRUE lemma must be ACCEPTED.
#   Negative -- a FALSE lemma must be REJECTED.
#
# The negative alone is not enough, and this is not hypothetical: the first
# version of this target had a scope error in its sentinel (`ℕ.+`, which is not
# valid syntax), so Agda rejected it for a reason that had nothing to do with
# the lemma being false -- and the target printed "the checker is live". A guard
# whose pass criterion is "the tool failed" passes whenever the guard is broken,
# which is the one failure mode a guard must not have.
redcheck: stdlib-check $(LIBFILE)
	@mkdir -p .redcheck
	@printf '{-# OPTIONS --safe #-}\nmodule SentinelTrue where\nopen import Data.Nat using (ℕ; _+_)\nopen import Relation.Binary.PropositionalEquality using (_≡_; refl)\ntrue-lemma : (2 + 2) ≡ 4\ntrue-lemma = refl\n' > .redcheck/SentinelTrue.agda
	@printf '{-# OPTIONS --safe #-}\nmodule SentinelFalse where\nopen import Data.Nat using (ℕ; _+_)\nopen import Relation.Binary.PropositionalEquality using (_≡_; refl)\nfalse-lemma : (2 + 2) ≡ 5\nfalse-lemma = refl\n' > .redcheck/SentinelFalse.agda
	@if ! $(AGDA) --library-file=$(LIBFILE) -i.redcheck .redcheck/SentinelTrue.agda >/dev/null 2>&1; then \
	  echo "POSITIVE SENTINEL REJECTED: the harness is broken, so a rejection below would mean nothing." >&2; \
	  $(AGDA) --library-file=$(LIBFILE) -i.redcheck .redcheck/SentinelTrue.agda 2>&1 | head -5 >&2; exit 1; \
	fi
	@if $(AGDA) --library-file=$(LIBFILE) -i.redcheck .redcheck/SentinelFalse.agda >/dev/null 2>&1; then \
	  echo "NEGATIVE SENTINEL ACCEPTED A FALSE LEMMA -- the checker is not checking." >&2; exit 1; \
	fi
	@echo "sentinels: true lemma accepted, false lemma rejected -- the checker is live"

clean:
	rm -rf _build .redcheck $(LIBFILE)
	find . -name '*.agdai' -delete
