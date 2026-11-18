"""Redact credential-like strings from model outputs before sharing them.

The model sometimes writes Alibaba Cloud AccessKeyIds (LTAI...), OSS signatures and tokens into the
URLs it generates. Some may be memorized real credentials. Never test them; redact them before
publishing. Rewrites every .json/.jsonl/.log/.txt file under the given paths in place and exits
non-zero if anything credential-like survives.

usage: python scripts/redact.py PATH [PATH ...]
"""
import os, re, sys

PATTERNS = [
    (re.compile(r"LTAI[0-9A-Za-z]{8,}"), "LTAI[REDACTED]"),
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{12,}"), r"\1[REDACTED]"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]*)?"), "[REDACTED-JWT]"),
    (re.compile(r"(?i)(Bearer\s+)(?!\[REDACTED)[A-Za-z0-9._~+/=-]{12,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(X-Amz-(?:Credential|Signature|Security-Token)=)(?!\[REDACTED\])[^&\s\"'\\]+"),
     r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:OSS)?AccessKey(?:Id|Secret)|Signature|x-oss-credential|x-oss-signature|"
                r"x-oss-security-token|security-token|SecurityToken|access_key_(?:id|secret))"
                r"(\\?[\"']?\s*[=:]\s*\\?[\"']?)(?!\[REDACTED\])[^&\s\"'\\,}]+"), r"\1\2[REDACTED]"),
]
CHECK = re.compile(r"LTAI(?!\[REDACTED\])[0-9A-Za-z]{8,}|\b(?:AKIA|ASIA)[0-9A-Z]{12,}|"
                   r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")


def files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        for d, _, fs in os.walk(p):
            for f in fs:
                if f.endswith((".json", ".jsonl", ".log", ".txt")):
                    yield os.path.join(d, f)


bad = 0
for f in files(sys.argv[1:]):
    s = open(f, encoding="utf-8", errors="surrogateescape").read()
    t = s
    for pat, rep in PATTERNS:
        t = pat.sub(rep, t)
    if t != s:
        open(f, "w", encoding="utf-8", errors="surrogateescape").write(t)
        print(f"redacted {f}")
    if CHECK.search(t):
        print(f"STILL CONTAINS KEY-LIKE STRING: {f}")
        bad += 1
sys.exit(1 if bad else 0)
