class Report:
    limit: int | None = 80 * 8
    label = "report"

    def cut(self, text: str) -> str:
        m = self.limit
        if m is None or len(text) <= m:
            return text
        return "%s: %d characters" % (self.label, len(text))


def render_all(reports: list[Report], text: str) -> list[str]:
    return [r.cut(text) for r in reports]
