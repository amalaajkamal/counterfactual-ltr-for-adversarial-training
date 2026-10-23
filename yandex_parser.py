"""
================================================================================
Yandex Personalized Web Search Challenge -- raw log parser
================================================================================
Parses the RAW train/test file format (M/Q/C lines), NOT the pre-parsed
query_id/position/doc_id/clicked CSV used earlier. This recovers what that
CSV was missing: per-click timestamps (TimePassed), the full SERP shown per
query (not just clicked items), and multi-query sessions -- enabling
dwell-time-derived relevance grades, following the OFFICIAL Yandex Challenge
grading scheme (see terms_english.pdf in your download):

    grade 0 (irrelevant):      no click, or click with dwell time < 50
    grade 1 (relevant):        click with dwell time in [50, 399]
    grade 2 (highly relevant): click with dwell time >= 400,
                                OR the click is the LAST action in the session
                                (regardless of measured dwell time)

--------------------------------------------------------------------------
RAW FORMAT (confirmed against real file lines, Sept 2026)
--------------------------------------------------------------------------
M line (session metadata):
    SessionID \t "M" \t TimePassed \t UserID

Q line (a query/SERP shown to the user):
    SessionID \t TimePassed \t "Q" \t SERPID \t QueryID \t [QueryTerms]
        \t URL1,Domain1 \t URL2,Domain2 \t ... \t URL10,Domain10
    QueryTerms is comma-separated term IDs; MAY BE ABSENT for some queries
    (distinguish it from URL,Domain pairs by: it has no comma-separated
    PAIR structure with exactly 2 numbers -- see _parse_q_line below).

C line (a click):
    SessionID \t TimePassed \t "C" \t SERPID \t URLID
    A click's SERPID links it back to a specific Q line within the same
    session (sessions CAN contain multiple queries/reformulations).

--------------------------------------------------------------------------
MEMORY SAFETY
--------------------------------------------------------------------------
The real train file is ~16.5GB / ~164M+ lines. This parser NEVER loads the
file into memory: it streams line-by-line and buffers only ONE session at a
time (sessions are contiguous in the file, bounded by consecutive M lines).
Each completed session is processed and yielded immediately, then discarded.

--------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------
process_yandex_file() yields one record per (session, SERP, shown document):
    {
        "session_id": int, "serp_id": int, "query_id": int, "user_id": int,
        "position": int (0-indexed, order shown in the Q line),
        "url_id": int, "domain_id": int,
        "clicked": 0/1,
        "dwell_time": int or None (None if not clicked),
        "relevance_grade": 0, 1, or 2,
    }
This is the natural unit for later model training: one row per document
shown in a SERP, exactly analogous to one row of the Yahoo!/MSLR libsvm
files (query, document, "relevance" label) -- except here the label is
click/dwell-time-derived, not editorial, and there are no content features
(only url_id/domain_id/query_id, meant to be fed through learned embedding
layers rather than a dense feature vector).
================================================================================
"""

import csv
import json
import sys
from collections import defaultdict


def _parse_m_line(parts):
    # SessionID  "M"  TimePassed  UserID
    return {"type": "M", "session_id": int(parts[0]), "time": int(parts[2]), "user_id": int(parts[3])}


def _parse_q_line(parts):
    # SessionID  TimePassed  "Q"  SERPID  QueryID  [QueryTerms]  URL,Domain x10
    session_id = int(parts[0])
    time = int(parts[1])
    serp_id = int(parts[3])
    query_id = int(parts[4])

    # Remaining fields are EITHER the (optional) query-terms field OR
    # URL,Domain pairs. A URL,Domain field always has exactly one comma
    # and both sides are plain integers; the query-terms field, when
    # present, is a comma-separated list of 1+ term IDs. Ambiguity is
    # possible only for a single-comma terms field vs a URL,Domain field --
    # resolved by checking whether the NEXT field also looks like a
    # URL,Domain pair (SERPs always list multiple documents; a lone
    # look-alike field before a run of URL,Domain fields is the terms field).
    rest = parts[5:]

    def looks_like_url_domain(field):
        pieces = field.split(",")
        return len(pieces) == 2 and all(p.isdigit() for p in pieces)

    query_terms = None
    doc_start_idx = 0
    if rest and not looks_like_url_domain(rest[0]):
        query_terms = rest[0]
        doc_start_idx = 1
    elif rest:
        # First field COULD be a 1-term query-terms field with no comma,
        # which is indistinguishable in shape from a bare integer -- but
        # URL,Domain fields always contain a comma, so a comma-less first
        # field is unambiguously query terms, not a document.
        if "," not in rest[0]:
            query_terms = rest[0]
            doc_start_idx = 1

    docs = []
    for field in rest[doc_start_idx:]:
        if not field:
            continue
        url_str, domain_str = field.split(",")
        docs.append((int(url_str), int(domain_str)))

    return {
        "type": "Q", "session_id": session_id, "time": time, "serp_id": serp_id,
        "query_id": query_id, "query_terms": query_terms, "docs": docs,
    }


def _parse_c_line(parts):
    # SessionID  TimePassed  "C"  SERPID  URLID
    return {
        "type": "C", "session_id": int(parts[0]), "time": int(parts[1]),
        "serp_id": int(parts[3]), "url_id": int(parts[4]),
    }


def _parse_line(line):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    record_type = parts[1] if parts[1] in ("M", "Q", "C") else (parts[2] if len(parts) > 2 else None)
    # M lines: type token is parts[1]. Q/C lines: type token is parts[2].
    if parts[1] == "M":
        return _parse_m_line(parts)
    if len(parts) > 2 and parts[2] == "Q":
        return _parse_q_line(parts)
    if len(parts) > 2 and parts[2] == "C":
        return _parse_c_line(parts)
    return None


def _relevance_grade(dwell_time, is_last_action_in_session):
    if is_last_action_in_session:
        return 2
    if dwell_time is None:
        return 0
    if dwell_time < 50:
        return 0
    if dwell_time < 400:
        return 1
    return 2


def _process_session(session_id, user_id, events):
    """
    events: chronologically-ordered list of Q/C dicts for one session.
    Computes dwell time per click as time-to-next-action (next Q or C in
    the session), with the session's final action always graded 2, per
    the official scheme. Returns one record per (SERP, shown document).
    """
    # index queries by serp_id (a session can show the same serp_id only
    # once in practice, but guard with a list in case of duplicates)
    queries_by_serp = defaultdict(list)
    for e in events:
        if e["type"] == "Q":
            queries_by_serp[e["serp_id"]].append(e)

    # clicks in chronological order (events list is already chronological)
    clicks = [e for e in events if e["type"] == "C"]

    # dwell time = time to next action (Q or C) after this click;
    # last action in the whole session gets no measured dwell time but
    # is graded 2 regardless (official rule)
    click_dwell = {}  # id(click_event) -> (dwell_time or None, is_last)
    for i, c in enumerate(clicks):
        # find this click's index among ALL events to get the true "next event"
        pass
    # simpler: walk the full chronological event list directly
    for i, e in enumerate(events):
        if e["type"] != "C":
            continue
        is_last = (i == len(events) - 1)
        if is_last:
            click_dwell[i] = (None, True)
        else:
            next_e = events[i + 1]
            dwell = next_e["time"] - e["time"]
            click_dwell[i] = (dwell, False)

    # clicked url_ids per serp_id, with their computed relevance grade
    clicked_grades = defaultdict(dict)  # serp_id -> {url_id: grade}
    for i, e in enumerate(events):
        if e["type"] != "C":
            continue
        dwell, is_last = click_dwell[i]
        grade = _relevance_grade(dwell, is_last)
        # if clicked multiple times, keep the highest grade observed
        prev = clicked_grades[e["serp_id"]].get(e["url_id"])
        if prev is None or grade > prev["relevance_grade"]:
            clicked_grades[e["serp_id"]][e["url_id"]] = {
                "relevance_grade": grade, "dwell_time": dwell,
            }

    records = []
    for serp_id, qlist in queries_by_serp.items():
        for q in qlist:
            for pos, (url_id, domain_id) in enumerate(q["docs"]):
                click_info = clicked_grades[serp_id].get(url_id)
                clicked = 1 if click_info is not None else 0
                grade = click_info["relevance_grade"] if click_info else 0
                dwell = click_info["dwell_time"] if click_info else None
                records.append({
                    "session_id": session_id, "user_id": user_id,
                    "serp_id": serp_id, "query_id": q["query_id"],
                    "position": pos, "url_id": url_id, "domain_id": domain_id,
                    "clicked": clicked, "dwell_time": dwell,
                    "relevance_grade": grade,
                })
    return records


def process_yandex_file(filepath, max_sessions=None, progress_every=100000,
                         max_parse_errors=None):
    """
    Streams the raw file, yielding one record dict per (session, SERP,
    shown document). Buffers only ONE session at a time -- safe for files
    of any size. Set max_sessions to bound the run (e.g., for a training
    subsample); None processes the entire file.

    Robustness: a malformed or unexpected line is logged and SKIPPED,
    not fatal -- a single bad row out of 164M+ should not crash a run
    that has already processed everything before it. Tracks a running
    error count; if max_parse_errors is set and exceeded, raises (since
    a very high error rate likely means something is systematically
    wrong with the file/encoding, not just a few bad rows).
    """
    current_session_id = None
    current_user_id = None
    current_events = []
    sessions_done = 0
    parse_errors = 0
    lines_seen = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            lines_seen = line_num
            if not line.strip():
                continue
            try:
                parsed = _parse_line(line)
            except Exception as e:
                parse_errors += 1
                if parse_errors <= 20 or parse_errors % 1000 == 0:
                    print(f"  [parse error #{parse_errors}] line {line_num}: {e} "
                          f"-- skipping. Line content: {line[:200]!r}", file=sys.stderr)
                if max_parse_errors is not None and parse_errors > max_parse_errors:
                    raise RuntimeError(
                        f"Exceeded max_parse_errors ({max_parse_errors}) at line {line_num}; "
                        f"this likely indicates a systematic problem (wrong file, encoding "
                        f"issue, unexpected format version) rather than a few bad rows."
                    )
                continue
            if parsed is None:
                continue

            try:
                if parsed["type"] == "M":
                    # flush the previous session before starting a new one
                    if current_session_id is not None and current_events:
                        for rec in _process_session(current_session_id, current_user_id, current_events):
                            yield rec
                        sessions_done += 1
                        if progress_every and sessions_done % progress_every == 0:
                            print(f"  ...processed {sessions_done} sessions "
                                  f"(line {line_num}, {parse_errors} parse errors so far)",
                                  file=sys.stderr)
                        if max_sessions is not None and sessions_done >= max_sessions:
                            return
                    current_session_id = parsed["session_id"]
                    current_user_id = parsed["user_id"]
                    current_events = []
                else:
                    current_events.append(parsed)
            except Exception as e:
                # a session-processing error (e.g., an oddly-shaped session)
                # should not kill the whole run either -- log it, drop this
                # session's partial state, and continue from the next M line.
                parse_errors += 1
                print(f"  [session error #{parse_errors}] near line {line_num}, "
                      f"session_id={current_session_id}: {e} -- dropping this session.",
                      file=sys.stderr)
                current_session_id = None
                current_events = []
                continue

    # flush the final session in the file
    if current_session_id is not None and current_events:
        try:
            for rec in _process_session(current_session_id, current_user_id, current_events):
                yield rec
        except Exception as e:
            print(f"  [session error] final session {current_session_id}: {e} -- dropped.",
                  file=sys.stderr)

    print(f"Done. {sessions_done} sessions processed, {lines_seen} lines read, "
          f"{parse_errors} parse/session errors encountered.", file=sys.stderr)


def write_subsample_csv(input_filepath, output_filepath, max_sessions=50000):
    """
    Convenience entry point: processes up to max_sessions sessions and
    writes the resulting (session, SERP, document) records to a CSV,
    suitable as input to a model with learned query_id/url_id/domain_id
    embeddings (no content features -- see module docstring).
    """
    fieldnames = ["session_id", "user_id", "serp_id", "query_id", "position",
                  "url_id", "domain_id", "clicked", "dwell_time", "relevance_grade"]
    n_written = 0
    with open(output_filepath, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in process_yandex_file(input_filepath, max_sessions=max_sessions):
            writer.writerow(rec)
            n_written += 1
    print(f"Wrote {n_written} (session, SERP, document) records to {output_filepath}")


if __name__ == "__main__":
    # Example usage -- edit paths as needed.
    INPUT_FILE = "yandex-personalized-web-search-challenge/train/train"
    OUTPUT_FILE = "yandex_subsample.csv"
    N_SESSIONS = 50000  # start small; scale up once this runs cleanly

    write_subsample_csv(INPUT_FILE, OUTPUT_FILE, max_sessions=N_SESSIONS)
