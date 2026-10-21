# WHAT: a package whose two modules import each other -- each one's
# annotations and bodies name the other's classes and fields -- used through
# the package: a case runs into a fresh result and into a shared one.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the bodies read fields of
# a class the other module declares, and which object each run fills is only
# visible in what the result holds. The import was refused as a "local source
# import cycle".
import a_package_whose_modules_import_each_other as pkg

shared = pkg.Result()
pkg.Case("a").run(shared)
pkg.Loud("b").run(shared)
print(shared.names())
print(pkg.Case("c").run().names())
print(isinstance(shared.seen[1], pkg.Loud))
