"""The readiness checker must itself be right: every check fires on a bad page and stays quiet on a good one."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_submission_readiness as cr  # noqa: E402


def site(tmp_path: Path, readme: str, receipts: list[dict] | None = None, files: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    res = tmp_path / "bench" / "results"
    res.mkdir(parents=True)
    if receipts is not None:
        (res / "receipts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in receipts), encoding="utf-8")
    for name, body in (files or {}).items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return tmp_path


def checks(root: Path, stage: str = "build", n: int | None = None) -> list[str]:
    return [f.check for f in cr.check_files(root, stage, n)]


EXTRACT_RECEIPT = {"stage": "extract", "on_amd": True, "ok": True, "host": "developer.amd.com.cn"}


def test_clean_page_passes(tmp_path: Path) -> None:
    root = site(tmp_path, "# SayAgain\n\nThe referee is plain Python. 12 tests.\n", [EXTRACT_RECEIPT])
    assert checks(root, n=12) == []


def test_placeholder_fails(tmp_path: Path) -> None:
    root = site(tmp_path, "Errors flagged: ⟨measured⟩. TODO: write this.\n")
    assert checks(root).count("placeholder") == 2


def test_unbacked_mi300x_fails_and_backed_passes(tmp_path: Path) -> None:
    root = site(tmp_path, "Runs on one MI300X.\n", [EXTRACT_RECEIPT])
    assert "hardware" in checks(root)
    (root / "bench" / "results" / "env.json").write_text(json.dumps({"gpu": "AMD Instinct MI300X"}))
    assert "hardware" not in checks(root)


def test_vllm_literal_needs_a_receipt(tmp_path: Path) -> None:
    root = site(tmp_path, "Served with vLLM.\n", [EXTRACT_RECEIPT])
    assert checks(root) == ["hardware"]


def test_amd_stage_mapping(tmp_path: Path) -> None:
    root = site(
        tmp_path, "Extraction runs on the AMD Token Factory.\nWhisper transcription runs on AMD.\n", [EXTRACT_RECEIPT]
    )
    found = cr.check_files(root, "build")
    assert [(f.check, f.line) for f in found] == [("amd-stage", 2)]


def test_amd_gpu_claim_needs_a_gpu_receipt(tmp_path: Path) -> None:
    root = site(tmp_path, "The model ran on AMD GPUs.\n", [EXTRACT_RECEIPT])
    assert checks(root) == ["amd-stage"]


def test_off_amd_sentence_is_allowed(tmp_path: Path) -> None:
    root = site(tmp_path, "ASR ran off-AMD on a laptop.\n", [])
    assert checks(root) == []


def test_negated_amd_sentence_is_not_a_claim(tmp_path: Path) -> None:
    root = site(tmp_path, "The AMD Token Factory serves no speech-to-text model.\nASR cannot run on AMD here.\n", [])
    assert checks(root) == []


def test_amd_without_any_receipt_fails(tmp_path: Path) -> None:
    root = site(tmp_path, "Built for the AMD hackathon. Everything runs on AMD.\n", [])
    assert checks(root) == ["amd-stage"]


def test_missing_results_file_fails(tmp_path: Path) -> None:
    root = site(tmp_path, "See bench/results/eval-0db.json for the table.\n")
    assert checks(root) == ["results"]


def test_test_count_must_match(tmp_path: Path) -> None:
    root = site(tmp_path, "Tested: 40 tests.\n")
    assert checks(root, n=41) == ["tests"]
    assert checks(root, n=40) == []


def test_submit_stage_requires_results_and_count(tmp_path: Path) -> None:
    root = site(tmp_path, "# x\n")
    assert sorted(checks(root, stage="submit")) == ["results", "results", "submit", "tests"]


def test_submit_stage_counts_the_text_rows_as_measured_results(tmp_path: Path) -> None:
    # the measured rows live in bench/results/text/ (no recordings yet); before, only eval-*.json counted
    root = site(
        tmp_path, "# x\n", files={"bench/results/text/ladder-r1.json": "{}", "bench/results/access-gate.json": "{}"}
    )
    assert sorted(checks(root, stage="submit")) == ["submit", "tests"]


def test_submit_stage_wants_the_live_replay_linked_and_nothing_unfinished(tmp_path: Path) -> None:
    files = {"bench/results/text/ladder-r1.json": "{}", "bench/results/access-gate.json": "{}"}
    stale = "Status: under construction. 3 tests.\nStatic replay (not published).\n"
    assert sorted(checks(site(tmp_path / "a", stale, files=files), stage="submit")) == ["submit", "submit", "submit"]
    done = "3 tests. Live replay: https://example.github.io/sayagain/replay/\n"
    assert checks(site(tmp_path / "b", done, files=files), stage="submit") == []
    assert checks(site(tmp_path / "c", stale, files=files), stage="build") == []


def test_old_hero_string_fails(tmp_path: Path) -> None:
    root = site(tmp_path, "The hero card: MISMATCH · line: B ≠ A.\n")
    assert checks(root) == ["hero"]
    root2 = site(tmp_path / "b", "The hero card: MISMATCH · tag: XV-203 ≠ XV-208.\n\nSecondary card: line: B ≠ A.\n")
    assert checks(root2) == []


def test_unbacked_number_fails_and_cited_passes(tmp_path: Path) -> None:
    root = site(
        tmp_path,
        "We flag 94% of errors.\nWe flag 94% of errors (bench/results/eval.json).\nTarget: ≥ 90 % flagged.\n"
        "We flag 97% (bench/results/eval.json).\n",
        files={"bench/results/eval.json": '{"flagged": 0.94}'},
    )
    found = cr.check_files(root, "build")
    assert [(f.check, f.line) for f in found] == [("number", 1), ("number", 4)]


def test_docs_folder_is_checked(tmp_path: Path) -> None:
    root = site(tmp_path, "# ok\n", files={"docs/DX-REPORT.md": "TBD\n"})
    assert checks(root) == ["placeholder"]


def test_repo_itself_passes_build_stage() -> None:
    assert cr.main(["--stage", "build", "--no-collect"]) == 0


def test_wider_claims_are_caught(tmp_path: Path) -> None:
    root = site(
        tmp_path,
        "Everything runs on AMD hardware, no cloud needed.\nSpeech recognition runs on AMD.\n"
        "Served on an AMD Radeon GPU.\nIt is 3x faster than by ear.\nLatency 1.8 s end to end.\n",
        [EXTRACT_RECEIPT],
    )
    assert sorted(checks(root)) == ["amd-stage", "amd-stage", "amd-stage", "hardware", "number", "number"]


def test_url_path_does_not_back_a_hardware_claim(tmp_path: Path) -> None:
    rec = {**EXTRACT_RECEIPT, "base_path": "/radeon/api/v1"}
    root = site(tmp_path, "Served on a Radeon card.\n", [rec])
    assert "hardware" in checks(root)


def test_every_public_page_is_scanned(tmp_path: Path) -> None:
    root = site(
        tmp_path,
        "# ok\n",
        files={"SECURITY.md": "TBD\n", "web/index.html": "<p>TODO</p>\n", "tests/notes.md": "TODO\n"},
    )
    assert checks(root) == ["placeholder", "placeholder"]


def test_stated_test_counts_anywhere_must_match(tmp_path: Path) -> None:
    root = site(tmp_path, "341 passing tests.\n", files={"docs/x.md": "We have 12 unit tests.\n"})
    assert checks(root, n=341) == ["tests"]
