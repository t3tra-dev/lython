# SystemExit(None) exits 0 in silence, as SystemExit() does; it printed
# "None" and exited 1.

print("before")
raise SystemExit(None)
