"""Local page for the support-triage decisions.

From the repository root:

    python code/view.py

Opens a loopback page that shows one card per ticket after the local BM25
pipeline runs. It writes under var/ and does not replace support_tickets/output.csv.
"""

import csv
import html
import sys
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.settings import default_config, repo_root
from data_loader.ticket_loader import load_tickets
from pipeline.run_pipeline import run_pipeline
from utils.logger import setup_logging


def _load_rows(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_page(tickets, rows) -> bytes:
    replied = sum(1 for row in rows if row["status"] == "replied")
    escalated = sum(1 for row in rows if row["status"] == "escalated")
    cards = []
    for index, (ticket, row) in enumerate(zip(tickets, rows), start=1):
        status = row["status"]
        cards.append(
            f"""
            <article class="card" data-status="{html.escape(status)}">
              <header>
                <span class="num">#{index}</span>
                <span class="pill {html.escape(status)}">{html.escape(status)}</span>
                <span class="meta">{html.escape(row["request_type"])}</span>
                <span class="meta">{html.escape(ticket.company or "unspecified")}</span>
              </header>
              <h2>{html.escape(ticket.subject or "No subject")}</h2>
              <p class="issue">{html.escape(ticket.issue)}</p>
              <p class="area">{html.escape(row["product_area"] or "No product area")}</p>
              <h3>{"Reply" if status == "replied" else "What the agent did"}</h3>
              <p>{html.escape(row["response"])}</p>
              <h3>Why</h3>
              <p class="why">{html.escape(row["justification"])}</p>
            </article>
            """
        )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support triage</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1c1915;
      --muted: #5e584f;
      --line: #e4ddd2;
      --paper: #f6f3ee;
      --card: #fffcf8;
      --replied: #1f6b45;
      --replied-bg: #e5f4eb;
      --escalated: #8a4b12;
      --escalated-bg: #f8ead8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background: var(--paper);
    }}
    main {{
      width: min(920px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0 72px;
    }}
    h1 {{
      font-size: 2.4rem;
      font-weight: 500;
      letter-spacing: -0.03em;
      margin: 0 0 8px;
    }}
    .lede {{
      max-width: 68ch;
      color: var(--muted);
      font-size: 1.08rem;
      line-height: 1.5;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 28px 0;
    }}
    .steps div {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      min-height: 108px;
    }}
    .steps strong {{ display: block; margin-bottom: 6px; }}
    .steps span {{ color: var(--muted); font-size: 0.95rem; line-height: 1.4; }}
    .bar {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin: 8px 0 18px;
    }}
    button {{
      font: inherit;
      border: 1px solid var(--line);
      background: var(--card);
      color: var(--ink);
      border-radius: 999px;
      padding: 6px 12px;
      cursor: pointer;
    }}
    button[aria-pressed="true"] {{
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 20px 8px;
      margin: 0 0 14px;
    }}
    .card header {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .num, .meta {{ color: var(--muted); font-size: 0.92rem; }}
    .pill {{
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .pill.replied {{ color: var(--replied); background: var(--replied-bg); }}
    .pill.escalated {{ color: var(--escalated); background: var(--escalated-bg); }}
    h2 {{ font-size: 1.35rem; font-weight: 500; margin: 10px 0 8px; }}
    h3 {{
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 16px 0 4px;
    }}
    p {{ line-height: 1.5; white-space: pre-wrap; }}
    .issue {{ color: var(--muted); }}
    .area {{ margin-top: 0; }}
    @media (max-width: 800px) {{
      .steps {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Support triage</h1>
    <p class="lede">
      This agent reads a support ticket, searches the local HackerRank, Claude, and Visa
      documentation, and either replies from those documents or escalates the ticket.
      Nothing is sent to an external model.
    </p>
    <section class="steps">
      <div><strong>1. Read</strong><span>Subject, message, and company from the ticket file.</span></div>
      <div><strong>2. Search</strong><span>BM25 ranks documentation chunks from the local corpus.</span></div>
      <div><strong>3. Check risk</strong><span>Account, security, legal, and outage language forces an escalation.</span></div>
      <div><strong>4. Decide</strong><span>A reply uses matching sentences. Weak or missing evidence escalates.</span></div>
    </section>
    <div class="bar">
      <button type="button" data-filter="all" aria-pressed="true">All {len(rows)}</button>
      <button type="button" data-filter="replied" aria-pressed="false">Replied {replied}</button>
      <button type="button" data-filter="escalated" aria-pressed="false">Escalated {escalated}</button>
    </div>
    {''.join(cards)}
  </main>
  <script>
    const buttons = document.querySelectorAll("button[data-filter]");
    const cards = document.querySelectorAll(".card");
    buttons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const filter = button.dataset.filter;
        buttons.forEach((item) => item.setAttribute("aria-pressed", item === button ? "true" : "false"));
        cards.forEach((card) => {{
          card.hidden = filter !== "all" && card.dataset.status !== filter;
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    return document.encode("utf-8")


def prepare() -> bytes:
    root = repo_root()
    output = root / "var" / "viewer-output.csv"
    log_path = root / "var" / "viewer-log.txt"
    defaults = replace(default_config(), output_path=output, log_path=log_path)
    setup_logging(log_path)
    run_pipeline(defaults, __import__("logging").getLogger("viewer"))
    tickets = load_tickets(defaults.input_path)
    rows = _load_rows(output)
    if len(tickets) != len(rows):
        raise RuntimeError("ticket count and triage rows do not match")
    return build_page(tickets, rows)


def serve(page: bytes, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            if self.path not in ("/", "/index.html"):
                body = b"not found"
                self.send_response(404)
            else:
                body = page
                self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(prepare())
