"""
MCP Server for Browser Form Testing with Ollama
"""

import asyncio
from typing import Dict, Optional, Any, List
from fastmcp import FastMCP
from browser_use import Agent, Browser
from langchain_ollama import ChatOllama

app = FastMCP("browser-use-form-mcp")


def _mk_agent(model: str, headless: bool, use_cloud: bool) -> Agent:
    """
    Create a browser-use Agent backed by Ollama.
    - headless=True for CI; set headless=False to watch the browser.
    - use_cloud=True if you have Browser Use Cloud; otherwise local Chromium.
    """
    browser = Browser(use_cloud=use_cloud, headless=headless)
    llm = ChatOllama(model=model)
    # We'll set the task later; agent supports setting task string at run()
    return Agent(llm=llm, browser=browser)


def _build_task(
    url: str,
    fields: Dict[str, str],
    submit_selector: Optional[str],
    submit_text: Optional[str],
    check_selector: Optional[str],
    must_contain_text: Optional[str],
    wait_after_submit_sec: int,
) -> str:
    """
    Turn structured inputs into a deterministic, step-by-step natural-language task
    for the browser-use agent. The agent will follow this plan using its tools.
    """
    lines: List[str] = []
    lines.append(f"Open {url}.")
    lines.append("Wait until the page is fully interactive.")
    
    # Filling strategy: try label/placeholder/name/id, then CSS, then fallbacks.
    lines.append("For each field below, find the input/textarea/select by (in order):")
    lines.append("1) a visible label matching the given key (case-insensitive),")
    lines.append("2) placeholder matching the key,")
    lines.append("3) name or id matching the key,")
    lines.append("4) or a CSS selector if the key looks like a selector (starts with '#' or '.' or contains '[').")
    lines.append("If multiple matches exist, choose the most visible and enabled one.")
    lines.append("Scroll into view before typing/selecting; do not paste invisible text.")
    lines.append("Fields to fill (key -> value):")
    for k, v in fields.items():
        safe_v = v.replace("\n", " ").strip()
        lines.append(f"- {k} -> {safe_v}")
    
    # Submit strategy
    if submit_selector:
        lines.append(f"Submit the form by clicking the element matching CSS selector: {submit_selector}.")
    elif submit_text:
        lines.append(f"Submit the form by clicking the most prominent button whose text contains: \"{submit_text}\" (case-insensitive).")
    else:
        lines.append("Submit the form by clicking the primary submit button (type=submit) within the same form as the last edited field.")
    
    # Post-submit checks
    lines.append(f"After submitting, wait up to {wait_after_submit_sec} seconds for navigation or DOM updates.")
    checks: List[str] = []
    if check_selector:
        checks.append(f"Confirm that a node matching CSS selector `{check_selector}` exists.")
    if must_contain_text:
        checks.append(f"Confirm the resulting page contains the text: \"{must_contain_text}\" (case-insensitive).")
    
    if checks:
        lines.append("Verification:")
        for c in checks:
            lines.append(f"- {c}")
        lines.append("If verification passes, output PASS and include a short explanation with any matched text.")
        lines.append("If verification fails, output FAIL and include a brief reason and any relevant visible error messages.")
    else:
        lines.append("Finally, summarize the result in one or two sentences.")
    
    # Compact final output
    lines.append("In your final note, include:")
    lines.append("- status: PASS or FAIL (if checks were requested; otherwise DONE)")
    lines.append("- final URL")
    lines.append("- page title")
    lines.append("- a one-paragraph summary")
    
    return "\n".join(lines)


@app.tool()
async def fill_form_and_check(
    url: str,
    fields: Dict[str, str],
    submit_selector: Optional[str] = None,
    submit_text: Optional[str] = None,
    check_selector: Optional[str] = None,
    must_contain_text: Optional[str] = None,
    model: str = "llama3.1:8b",
    headless: bool = True,
    use_cloud_browser: bool = False,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """
    Open a URL, fill a form, submit, and verify the result.
    
    Args:
      url: Page containing the form.
      fields: Mapping of field keys to values. Keys can be labels ('Email'), names/ids,
              placeholders, or CSS selectors ('#email', '.email-input', 'input[name=email]').
      submit_selector: CSS selector for the submit trigger (preferred for determinism).
      submit_text: Fallback button text to click if no selector provided (e.g., 'Sign in').
      check_selector: CSS selector that must appear after submit to count as PASS.
      must_contain_text: Text that must appear after submit (case-insensitive) to count as PASS.
      model: Ollama model id (e.g., 'llama3.1:8b').
      headless: Run browser headless (True) or visible (False).
      use_cloud_browser: Use Browser Use Cloud instead of local browser.
      timeout_seconds: Total timeout for the end-to-end task.
    
    Returns:
      status: "ok" (tool ran) and an embedded result: PASS/FAIL/DONE + details
      steps: a compact execution trace from browser-use
      final: the agent's final note (includes status/URL/title/summary)
    """
    agent = _mk_agent(model=model, headless=headless, use_cloud=use_cloud_browser)
    
    task = _build_task(
        url=url,
        fields=fields,
        submit_selector=submit_selector,
        submit_text=submit_text,
        check_selector=check_selector,
        must_contain_text=must_contain_text,
        wait_after_submit_sec=45,
    )
    
    try:
        history = await asyncio.wait_for(agent.run(task=task), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"status": "ok", "result": "TIMEOUT", "final": f"Timed out after {timeout_seconds}s", "steps": []}
    except Exception as e:
        return {"status": "ok", "result": "ERROR", "final": str(e), "steps": []}
    
    # Summarize steps defensively
    steps: List[str] = []
    for i, h in enumerate(history):
        try:
            action = getattr(h, "action", None) or (h.get("action") if isinstance(h, dict) else None)
            note = getattr(h, "note", None) or (h.get("note") if isinstance(h, dict) else None)
            steps.append(f"Step {i+1}: {action or 'action'} — {(note or '')[:180]}")
        except Exception:
            steps.append(f"Step {i+1}: {str(h)[:200]}")
    
    final_note = ""
    if history:
        last = history[-1]
        try:
            final_note = getattr(last, "note", None) or (last.get("note") if isinstance(last, dict) else "")
        except Exception:
            final_note = str(last)[:1000]
    
    # Try to extract PASS/FAIL/DONE from final note
    result = "DONE"
    for tag in ("PASS", "FAIL", "DONE"):
        if tag in (final_note or "").upper():
            result = tag
            break
    
    return {
        "status": "ok",
        "result": result,
        "final": final_note,
        "steps": steps[:40],
        "model": model,
    }


if __name__ == "__main__":
    app.run()
