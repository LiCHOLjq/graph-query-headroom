# Completed offline checks

`reproduction_report.json` records a complete local execution of the documented
entry point with all six analyses, raw-record checks, and 40 generated reference
CSV comparisons. The feature-availability models were refitted from numerical
costs and cached scalar features. It completed in approximately 14 seconds in
the recorded Python environment; runtime on another machine can differ.

`raw_audit.json` independently records checks of repeated-call medians, policy
ordering, scalar audit flags, and preserved censoring. It does not independently
reconstruct graph eligibility or verify answer IDs omitted from this release.

`release_audit.json` records the scope, extension/schema review, and text scans
including decompressed timing CSVs. No matching credentials, private keys,
personal account paths, or email addresses were found. This is a bounded audit,
not a guarantee against every possible disclosure risk.

The local preparation included an initial successful execution of all analyses
whose final comparator found four reference tables produced by other, excluded
entry points. Those four tables were removed from this narrower release; no
measurements or reported values were altered. The final complete run passed.

The original candidate remains untouched. The release retains source-member
hashes and describes each field projection. The one-time extraction helper is
not part of the public package; no claim is made that the public package can
reconstruct omitted private/upstream files. Its supported reproduction begins
from the complete numerical inputs included here.
