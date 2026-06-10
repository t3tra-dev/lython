#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace lython::emitter::typing {

struct Reason {
  std::string message;
};

struct Term {
  enum class Kind { Var, Con, Func };

  Kind kind = Kind::Con;
  std::uint64_t id = 0;
  std::string name;
  std::vector<Term> args;
  std::vector<Term> kwonly;
  std::vector<Term> results;

  static Term var(std::uint64_t id, llvm::StringRef name = {});
  static Term con(llvm::StringRef name, llvm::ArrayRef<Term> args = {});
  static Term func(llvm::ArrayRef<Term> args, llvm::ArrayRef<Term> results,
                   llvm::ArrayRef<Term> kwonly = {});

  std::string display() const;
};

bool operator==(const Term &lhs, const Term &rhs);
bool operator!=(const Term &lhs, const Term &rhs);

struct Scheme {
  std::set<std::uint64_t> variables;
  Term body;
};

class Error : public std::runtime_error {
public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

class Oracle {
public:
  virtual ~Oracle() = default;

  virtual std::optional<Term> attribute(const Term &owner,
                                        llvm::StringRef name) const;
  virtual std::optional<Term> call(const Term &callee,
                                   llvm::ArrayRef<Term> args) const;
  virtual std::optional<Term> subscript(const Term &container,
                                        const Term &index) const;
  virtual std::optional<Term> awaitable(const Term &awaitable) const;
};

class Environment {
public:
  void push();
  void pop();
  void bind(llvm::StringRef name, Scheme scheme);
  const Scheme &lookup(llvm::StringRef name) const;
  std::set<std::uint64_t> freeVars() const;

private:
  std::vector<std::map<std::string, Scheme>> scopes{{}};
};

class AlgorithmM {
public:
  explicit AlgorithmM(const Oracle &oracle);

  Term fresh(llvm::StringRef name = {});
  Scheme scheme(Term term, std::set<std::uint64_t> variables = {}) const;
  void bind(llvm::StringRef name, Term term, bool generalized = false);
  Term lookup(llvm::StringRef name);
  Scheme generalize(const Term &term) const;
  Term instantiate(const Scheme &scheme);

  void requireEqual(Term lhs, Term rhs, Reason reason);
  void requireAttribute(Term owner, llvm::StringRef name, Term result,
                        Reason reason);
  void requireCall(Term callee, llvm::ArrayRef<Term> args, Term result,
                   Reason reason);
  void requireSubscript(Term container, Term index, Term result, Reason reason);
  void requireAwait(Term awaitable, Term result, Reason reason);

  void solve();
  Term resolve(const Term &term);
  Term apply(const Term &term) const;
  void unify(Term lhs, Term rhs, std::optional<Reason> reason = std::nullopt);

private:
  struct Constraint {
    enum class Kind { Equal, Attribute, Call, Subscript, Await };

    Kind kind = Kind::Equal;
    Term lhs;
    Term rhs;
    std::vector<Term> args;
    std::string name;
    Reason reason;
  };

  const Oracle &oracle;
  Environment env;
  std::vector<Constraint> constraints;
  std::map<std::uint64_t, Term> substitution;
  std::uint64_t nextTypeVar = 0;

  Constraint apply(const Constraint &constraint) const;
  void bindVar(const Term &var, const Term &term, std::optional<Reason> reason);
  [[noreturn]] void failUnify(const Term &lhs, const Term &rhs,
                              std::optional<Reason> reason) const;
};

std::set<std::uint64_t> freeVars(const Term &term);
Term applySubstitution(const Term &term,
                       const std::map<std::uint64_t, Term> &substitution);

} // namespace lython::emitter::typing
