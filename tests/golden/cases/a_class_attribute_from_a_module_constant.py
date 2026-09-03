# A class body reading a module-level name. The unannotated spelling of the
# constant has to reach the class attribute the same way the annotated one
# does -- the attribute is typed before any body is emitted, and an attribute
# whose type is unknown has no storage to be read out of.

WIDTH = 80
LABEL = "grid"
SCALE = 1.5

ROWS: int = 4


class Layout:
    half = WIDTH // 2
    name = LABEL + "-layout"
    step = SCALE * 2
    cells = WIDTH * ROWS
    edges = [WIDTH, ROWS]


print(Layout.half, Layout.name, Layout.step, Layout.cells)
print(Layout.edges)


def report(layout: Layout) -> str:
    return layout.name + ":" + str(layout.half)


print(report(Layout()))
